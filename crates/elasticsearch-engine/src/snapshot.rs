//! Snapshot repositories (`type: fs`) and snapshots.
//!
//! A repository is a directory under `<data>/repo/<location>`; a snapshot is
//! `snap-<name>.json` (metadata) plus `snap-<name>/<index>/` (the committed
//! index directory, hard-linked where immutable). Restore copies an index
//! directory back under `<data>/indices/<uuid>` and registers it.

use crate::error::EsError;
use crate::index::{new_uuid, IndexService};
use crate::mapping::wildcard_match;
use crate::node::{Node, ResolveOpts, NODE_ID};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Result<T> = std::result::Result<T, EsError>;

// `SnapshotInfo` renders `IndexVersion.CURRENT`: its id and release range
// (`IndexVersions.VERSION_LOOKUP`: 9111000 spans 9.5.2..9.5.5).
const SNAPSHOT_VERSION: &str = "9.5.2-9.5.5";
const SNAPSHOT_VERSION_ID: i64 = 9_111_000;

#[derive(Default)]
pub struct SnapshotService {
    data: PathBuf,
    repos: RwLock<BTreeMap<String, Value>>,
    /// Coordinator metadata used only for registration validation, never filesystem operations.
    cluster_repos: RwLock<Option<BTreeMap<String, Value>>>,
    /// index name -> (start millis, stop millis, recovered bytes, files)
    restored: Mutex<HashMap<String, (i64, i64, u64, u64)>>,
    op_lock: Mutex<()>,
}

pub fn repo_missing(name: &str) -> EsError {
    EsError::new(404, "repository_missing_exception", format!("[{name}] missing"))
}

pub fn snapshot_missing(repo: &str, snap: &str) -> EsError {
    EsError::new(404, "snapshot_missing_exception", format!("[{repo}:{snap}] is missing"))
}

fn matches_any(pats: &[String], name: &str) -> bool {
    pats.iter().any(|p| p == "_all" || p == "*" || (p.contains('*') && wildcard_match(p, name)) || p == name)
}

fn dir_files(p: &Path) -> (u64, u64) {
    let (mut n, mut b) = (0, 0);
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                let (fn_, fb) = dir_files(&path);
                n += fn_;
                b += fb;
            } else {
                n += 1;
                b += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    (n, b)
}

impl SnapshotService {
    pub fn load(data: &Path) -> SnapshotService {
        let repos: BTreeMap<String, Value> = std::fs::read(data.join("_state/repositories.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        SnapshotService { data: data.to_path_buf(), repos: RwLock::new(repos), ..Default::default() }
    }

    fn persist(&self) -> Result<()> {
        let _ = std::fs::create_dir_all(self.data.join("_state"));
        let tmp = self.data.join("_state/repositories.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&*self.repos.read())?)?;
        std::fs::rename(tmp, self.data.join("_state/repositories.json"))?;
        Ok(())
    }

    fn repo_dir(&self, repo: &Value) -> PathBuf {
        let loc = repo["settings"]["location"].as_str().unwrap_or("");
        self.data.join("repo").join(loc.trim_start_matches('/'))
    }

    fn dir_of(&self, name: &str) -> Result<PathBuf> {
        let repos = self.repos.read();
        let r = repos.get(name).ok_or_else(|| repo_missing(name))?;
        Ok(self.repo_dir(r))
    }

    /// Directory of repository `name` (fs repositories).
    pub fn repository_dir(&self, name: &str) -> Result<PathBuf> {
        self.dir_of(name)
    }

    /// A setting of repository `name` (`None` when unset or no such repository).
    pub fn repository_setting(&self, name: &str, key: &str) -> Option<Value> {
        self.repos.read().get(name).map(|r| r["settings"][key].clone()).filter(|v| !v.is_null())
    }

    pub fn restore_cluster_registrations(&self, repositories: BTreeMap<String, Value>) {
        *self.cluster_repos.write() = Some(repositories);
    }

    /// Registration and readonly status from the authoritative coordinator catalog.
    pub fn registered_readonly(&self, name: &str) -> Option<bool> {
        let cluster = self.cluster_repos.read();
        let local = self.repos.read();
        let repositories = cluster.as_ref().unwrap_or(&local);
        repositories.get(name).map(|r| {
            ["readonly", "read_only"].iter().any(|key| {
                r["settings"][*key] == Value::Bool(true) || r["settings"][*key].as_str() == Some("true")
            })
        })
    }

    pub fn has_repository(&self, name: &str) -> bool {
        self.repos.read().contains_key(name)
    }

    pub fn is_read_only(&self, name: &str) -> bool {
        self.repos.read().contains_key(name) && self.read_only(name)
    }

    fn read_only(&self, name: &str) -> bool {
        let repos = self.repos.read();
        let s = &repos[name]["settings"];
        ["readonly", "read_only"].iter().any(|k| matches!(&s[*k], Value::Bool(true)) || s[*k].as_str() == Some("true"))
    }

    // ---- repositories ------------------------------------------------

    pub fn put_repository(&self, name: &str, body: &Value) -> Result<()> {
        let ty = body["type"].as_str().ok_or_else(|| EsError::action_validation(&["type is missing"]))?;
        if ty != "fs" && ty != "source" {
            return Err(EsError::new(500, "repository_exception", format!("[{name}] repository type [{ty}] does not exist"))
                .caused_by(EsError::new(400, "illegal_argument_exception", format!("Unknown [repository] type [{ty}]"))));
        }
        let settings = body.get("settings").cloned().unwrap_or_else(|| json!({}));
        let loc = settings["location"].as_str().unwrap_or("");
        if loc.is_empty() {
            return Err(EsError::new(500, "repository_exception", format!("[{name}] failed to create repository"))
                .caused_by(EsError::new(500, "repository_exception", format!("[{name}] missing location"))));
        }
        if loc.split('/').any(|c| c == "..") {
            return Err(EsError::new(500, "repository_exception", format!("[{name}] location [{loc}] doesn't match any of the locations specified by path.repo")));
        }
        let mut settings = settings;
        if let Some(o) = settings.as_object_mut() {
            for v in o.values_mut() {
                if !v.is_string() {
                    let t = match &*v {
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    };
                    *v = Value::String(t);
                }
            }
        }
        let mut repo = json!({"type": ty, "settings": settings});
        let dir = self.repo_dir(&repo);
        std::fs::create_dir_all(&dir)?;
        // The repository uuid lives in the location, so repositories sharing one agree.
        let uuid_file = dir.join("_repo_uuid");
        let uuid = match std::fs::read_to_string(&uuid_file) {
            Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                let u = new_uuid();
                let _ = std::fs::write(&uuid_file, &u);
                u
            }
        };
        repo["uuid"] = json!(uuid);
        self.repos.write().insert(name.to_string(), repo);
        self.persist()
    }

    pub fn delete_repository(&self, expr: &str) -> Result<()> {
        let pats: Vec<String> = expr.split(',').map(str::to_string).collect();
        let names: Vec<String> = self.repos.read().keys().filter(|k| matches_any(&pats, k)).cloned().collect();
        if names.is_empty() {
            return Err(repo_missing(expr));
        }
        for n in names {
            self.repos.write().remove(&n);
        }
        self.persist()
    }

    pub fn get_repositories(&self, expr: Option<&str>) -> Result<Value> {
        let repos = self.repos.read();
        let pats: Vec<String> = match expr {
            None | Some("") => vec!["_all".into()],
            Some(e) => e.split(',').map(str::to_string).collect(),
        };
        let mut out = Map::new();
        for (k, v) in repos.iter() {
            if matches_any(&pats, k) {
                out.insert(k.clone(), v.clone());
            }
        }
        for p in &pats {
            if !p.contains('*') && p != "_all" && !repos.contains_key(p) {
                return Err(repo_missing(p));
            }
        }
        if out.is_empty() && !pats.iter().any(|p| p.contains('*') || p == "_all") {
            return Err(repo_missing(expr.unwrap_or("_all")));
        }
        Ok(Value::Object(out))
    }

    /// Every registered repository: on a cluster node, the published
    /// catalog (each node knows every repository, as in ES), else this
    /// node's own registrations.
    pub fn repository_names(&self) -> Vec<String> {
        match self.cluster_repos.read().as_ref() {
            Some(cluster) => cluster.keys().cloned().collect(),
            None => self.repos.read().keys().cloned().collect(),
        }
    }

    pub fn verify(&self, name: &str, node_name: &str) -> Result<Value> {
        self.dir_of(name)?;
        Ok(json!({"nodes": {NODE_ID: {"name": node_name}}}))
    }

    // ---- snapshots ---------------------------------------------------

    fn meta_path(dir: &Path, snap: &str) -> PathBuf {
        dir.join(format!("snap-{snap}.json"))
    }

    fn list_names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        let n = e.file_name().to_string_lossy().into_owned();
                        n.strip_prefix("snap-").and_then(|s| s.strip_suffix(".json")).map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    fn read_meta(dir: &Path, repo: &str, snap: &str) -> Result<Value> {
        let b = std::fs::read(Self::meta_path(dir, snap)).map_err(|_| snapshot_missing(repo, snap))?;
        Ok(serde_json::from_slice(&b)?)
    }

    /// Snapshot names matching a comma separated expression.
    fn resolve_names(dir: &Path, repo: &str, expr: &str, ignore_unavailable: bool) -> Result<Vec<String>> {
        let all = Self::list_names(dir);
        let pats: Vec<String> = expr.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
        let mut out = Vec::new();
        for p in &pats {
            if p == "_current" {
                continue;
            }
            if p == "_all" || p == "*" || p.contains('*') {
                for n in &all {
                    if (p == "_all" || wildcard_match(p, n)) && !out.contains(n) {
                        out.push(n.clone());
                    }
                }
            } else if all.contains(p) {
                if !out.contains(p) {
                    out.push(p.clone());
                }
            } else if !ignore_unavailable {
                return Err(snapshot_missing(repo, p));
            }
        }
        Ok(out)
    }

    pub fn create_snapshot(&self, node: &Node, repo: &str, snap: &str, body: &Value) -> Result<Value> {
        let _g = self.op_lock.lock();
        let dir = self.dir_of(repo)?;
        if self.read_only(repo) {
            return Err(EsError::new(500, "repository_exception", format!("[{repo}] cannot create snapshot in a readonly repository")));
        }
        let snap = crate::node::resolve_date_math(snap)?;
        if snap.is_empty() || snap.starts_with('_') || snap.contains(['/', '\\', ' ', ',', '"', '*', '?', '<', '>', '|', '#']) || snap != snap.to_lowercase() {
            return Err(EsError::new(400, "invalid_snapshot_name_exception", format!("[{repo}:{snap}] Invalid snapshot name [{snap}], must be lowercase and not contain special characters")));
        }
        if Self::meta_path(&dir, &snap).exists() {
            return Err(EsError::new(400, "invalid_snapshot_name_exception", format!("[{repo}:{snap}] Invalid snapshot name [{snap}], snapshot with the same name already exists")));
        }
        let ignore_unavailable = body["ignore_unavailable"].as_bool().unwrap_or(false);
        let expr = match &body["indices"] {
            Value::Null => "*".to_string(),
            Value::String(s) if s.is_empty() => "*".to_string(),
            v => crate::node::pattern_list(v).join(","),
        };
        let mut opts = ResolveOpts { ignore_unavailable, allow_no_indices: true, expand_open: true, expand_closed: false, expand_hidden: true };
        if expr.split(',').any(|p| p == "none" || p == "_none") {
            opts.ignore_unavailable = true;
        }
        let targets = if expr == "none" { Vec::new() } else { node.resolve(&expr, opts)? };
        let start = crate::date::now_millis();
        let snap_dir = dir.join(format!("snap-{snap}"));
        let source_only = self.repos.read().get(repo).is_some_and(|r| r["type"] == "source");
        let mut indices: Vec<String> = Vec::new();
        let mut details = Map::new();
        let mut total_shards = 0usize;
        let mut failed_shards = 0usize;
        let mut failures: Vec<Value> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for t in &targets {
            let svc = &t.svc;
            if !seen.insert(svc.name.clone()) {
                continue;
            }
            if !svc.is_open() {
                continue;
            }
            let shard_count = svc.setting_usize("index.number_of_shards", 1);
            total_shards += shard_count;
            indices.push(svc.name.clone());
            // A `source` repository snapshots `_source` only: an index whose
            // source is incomplete fails every shard (`SourceOnlySnapshotRepository`).
            if source_only && !has_complete_source(svc) {
                for shard in 0..shard_count {
                    failures.push(json!({
                        "index": svc.name, "index_uuid": svc.uuid, "shard_id": shard,
                        "reason": "IllegalStateException[Can't snapshot _source only on an index that has incomplete source ie. has _source disabled or filters the source]",
                        "node_id": NODE_ID, "status": "INTERNAL_SERVER_ERROR",
                    }));
                }
                failed_shards += shard_count;
                continue;
            }
            copy_index(svc, &snap_dir.join(&svc.name))?;
            if source_only {
                mark_source_only(&snap_dir.join(&svc.name))?;
            }
            let (files, bytes) = dir_files(&snap_dir.join(&svc.name));
            details.insert(svc.name.clone(), json!({"shard_count": shard_count, "size_in_bytes": bytes, "file_count": files, "max_segments_per_shard": 1}));
        }
        let end = crate::date::now_millis();
        let include_global_state = body["include_global_state"].as_bool().unwrap_or(true);
        // Global metadata: persistent cluster settings and templates.
        let global_state = if include_global_state {
            let t = node.templates.read();
            json!({
                "persistent_settings": node.cluster_settings.read().0,
                "legacy_templates": t.legacy, "index_templates": t.index, "component_templates": t.component,
            })
        } else {
            Value::Null
        };
        let meta = json!({
            "snapshot": snap, "uuid": new_uuid(), "repository": repo,
            "version_id": SNAPSHOT_VERSION_ID, "version": SNAPSHOT_VERSION,
            "indices": indices, "data_streams": [],
            "include_global_state": include_global_state,
            "metadata": body.get("metadata").cloned().unwrap_or(Value::Null),
            "state": if failed_shards > 0 { "PARTIAL" } else { "SUCCESS" },
            "start_time_in_millis": start, "end_time_in_millis": end,
            "failures": failures, "shards": {"total": total_shards, "failed": failed_shards, "successful": total_shards - failed_shards},
            "index_details": details,
            "global_state": global_state,
        });
        std::fs::create_dir_all(&dir)?;
        std::fs::write(Self::meta_path(&dir, &snap), serde_json::to_vec_pretty(&meta)?)?;
        Ok(meta)
    }

    /// Snapshot metadata as the REST `info` object.
    pub fn info_json(meta: &Value, verbose: bool, index_details: bool, index_names: bool, include_repository: bool) -> Value {
        let mut o = Map::new();
        o.insert("snapshot".into(), meta["snapshot"].clone());
        o.insert("uuid".into(), meta["uuid"].clone());
        if include_repository {
            o.insert("repository".into(), meta["repository"].clone());
        }
        if verbose {
            o.insert("version_id".into(), meta["version_id"].clone());
            o.insert("version".into(), meta["version"].clone());
        }
        if index_names {
            o.insert("indices".into(), meta["indices"].clone());
        }
        o.insert("data_streams".into(), meta["data_streams"].clone());
        o.insert("include_global_state".into(), meta["include_global_state"].clone());
        if verbose && !meta["metadata"].is_null() {
            o.insert("metadata".into(), meta["metadata"].clone());
        }
        o.insert("state".into(), meta["state"].clone());
        if index_details {
            o.insert("index_details".into(), meta["index_details"].clone());
        }
        if verbose {
            let (s, e) = (meta["start_time_in_millis"].as_i64().unwrap_or(0), meta["end_time_in_millis"].as_i64().unwrap_or(0));
            let iso = |ms: i64| crate::date::from_millis(ms).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
            o.insert("start_time".into(), json!(iso(s)));
            o.insert("start_time_in_millis".into(), json!(s));
            o.insert("end_time".into(), json!(iso(e)));
            o.insert("end_time_in_millis".into(), json!(e));
            o.insert("duration_in_millis".into(), json!(e - s));
            o.insert("failures".into(), meta["failures"].clone());
            o.insert("shards".into(), meta["shards"].clone());
        }
        o.insert("feature_states".into(), json!([]));
        Value::Object(o)
    }

    /// Raw metadata of the snapshots matching `expr` in `repo`.
    pub fn get_snapshots(&self, repo: &str, expr: &str, ignore_unavailable: bool) -> Result<Vec<Value>> {
        let dir = self.dir_of(repo)?;
        let names = Self::resolve_names(&dir, repo, expr, ignore_unavailable)?;
        names.iter().map(|n| Self::read_meta(&dir, repo, n)).collect()
    }

    pub fn delete_snapshots(&self, repo: &str, expr: &str) -> Result<()> {
        let _g = self.op_lock.lock();
        let dir = self.dir_of(repo)?;
        if self.read_only(repo) {
            return Err(EsError::new(500, "repository_exception", format!("[{repo}] cannot delete snapshot from a readonly repository")));
        }
        let names = Self::resolve_names(&dir, repo, expr, false)?;
        for n in names {
            let _ = std::fs::remove_file(Self::meta_path(&dir, &n));
            let _ = std::fs::remove_dir_all(dir.join(format!("snap-{n}")));
        }
        Ok(())
    }

    pub fn clone_snapshot(&self, repo: &str, snap: &str, target: &str, body: &Value) -> Result<()> {
        let _g = self.op_lock.lock();
        let dir = self.dir_of(repo)?;
        let mut meta = Self::read_meta(&dir, repo, snap)?;
        if Self::meta_path(&dir, target).exists() {
            return Err(EsError::new(400, "invalid_snapshot_name_exception", format!("[{repo}:{target}] Invalid snapshot name [{target}], snapshot with the same name already exists")));
        }
        let pats = crate::node::pattern_list(&body["indices"]);
        if pats.is_empty() {
            return Err(EsError::new(400, "action_request_validation_exception", "Validation Failed: 1: indices is empty;"));
        }
        let have: Vec<String> = meta["indices"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let chosen: Vec<String> = have.iter().filter(|i| matches_any(&pats, i)).cloned().collect();
        for p in &pats {
            if !p.contains('*') && p != "_all" && !have.contains(p) {
                return Err(EsError::index_not_found(p));
            }
        }
        let src = dir.join(format!("snap-{snap}"));
        let dst = dir.join(format!("snap-{target}"));
        let mut details = Map::new();
        for i in &chosen {
            crate::ilm::resize::copy_tree(&src.join(i), &dst.join(i))?;
            if let Some(d) = meta["index_details"].get(i) {
                details.insert(i.clone(), d.clone());
            }
        }
        let now = crate::date::now_millis();
        meta["snapshot"] = json!(target);
        meta["uuid"] = json!(new_uuid());
        meta["indices"] = json!(chosen);
        let cloned_shards: u64 = details.values().map(|d| d["shard_count"].as_u64().unwrap_or(1)).sum();
        meta["index_details"] = Value::Object(details);
        meta["shards"] = json!({"total": cloned_shards, "failed": 0, "successful": cloned_shards});
        meta["start_time_in_millis"] = json!(now);
        meta["end_time_in_millis"] = json!(now);
        std::fs::write(Self::meta_path(&dir, target), serde_json::to_vec_pretty(&meta)?)?;
        Ok(())
    }

    /// `_status`: per snapshot shard statistics.
    pub fn status_json(&self, repo: &str, meta: &Value) -> Result<Value> {
        let dir = self.dir_of(repo)?;
        let snap = meta["snapshot"].as_str().unwrap_or("");
        let snap_dir = dir.join(format!("snap-{snap}"));
        let (mut files, mut bytes) = (0u64, 0u64);
        let mut indices = Map::new();
        for i in meta["indices"].as_array().cloned().unwrap_or_default() {
            let name = i.as_str().unwrap_or("");
            let (f, b) = dir_files(&snap_dir.join(name));
            files += f;
            bytes += b;
            let st = json!({"incremental": {"file_count": f, "size_in_bytes": b}, "total": {"file_count": f, "size_in_bytes": b},
                            "start_time_in_millis": meta["start_time_in_millis"], "time_in_millis": 0});
            indices.insert(
                name.to_string(),
                json!({"shards_stats": {"initializing": 0, "started": 0, "finalizing": 0, "done": 1, "failed": 0, "total": 1},
                       "stats": st, "shards": {"0": {"stage": "DONE", "stats": st}}}),
            );
        }
        let (s, e) = (meta["start_time_in_millis"].as_i64().unwrap_or(0), meta["end_time_in_millis"].as_i64().unwrap_or(0));
        let n = meta["indices"].as_array().map(|a| a.len()).unwrap_or(0);
        Ok(json!({
            "snapshot": meta["snapshot"], "repository": repo, "uuid": meta["uuid"], "state": meta["state"],
            "include_global_state": meta["include_global_state"],
            "shards_stats": {"initializing": 0, "started": 0, "finalizing": 0, "done": n, "failed": 0, "total": n},
            "stats": {"incremental": {"file_count": files, "size_in_bytes": bytes}, "total": {"file_count": files, "size_in_bytes": bytes},
                      "start_time_in_millis": s, "time_in_millis": e - s},
            "indices": indices,
        }))
    }

    // ---- restore -----------------------------------------------------

    pub fn restore(&self, node: &Node, repo: &str, snap: &str, body: &Value) -> Result<Value> {
        let _g = self.op_lock.lock();
        let dir = self.dir_of(repo)?;
        let meta = Self::read_meta(&dir, repo, snap)?;
        let have: Vec<String> = meta["indices"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let pats = match &body["indices"] {
            Value::Null => vec!["*".to_string()],
            v => {
                let p = crate::node::pattern_list(v);
                if p.is_empty() { vec!["*".to_string()] } else { p }
            }
        };
        let ignore_unavailable = body["ignore_unavailable"].as_bool().unwrap_or(false);
        for p in &pats {
            if !p.contains('*') && p != "_all" && !have.contains(p) && !ignore_unavailable {
                return Err(EsError::new(500, "snapshot_restore_exception", format!("[{repo}:{snap}] index [{p}] not found in snapshot")));
            }
        }
        let chosen: Vec<String> = have.iter().filter(|i| matches_any(&pats, i)).cloned().collect();
        let rename = |n: &str| -> Result<String> {
            match (body["rename_pattern"].as_str(), body["rename_replacement"].as_str()) {
                (Some(p), Some(r)) => {
                    let re = regex::Regex::new(p).map_err(|e| EsError::illegal_argument(format!("rename_pattern invalid: {e}")))?;
                    let r = regex::Regex::new(r"\$(\d+)").unwrap().replace_all(r, "$${$1}").into_owned();
                    Ok(re.replace_all(n, r.as_str()).into_owned())
                }
                _ => Ok(n.to_string()),
            }
        };
        // Validate first so a failure restores nothing.
        let mut plan = Vec::new();
        let partial = body["partial"].as_bool().unwrap_or(false);
        for i in &chosen {
            let failed = meta["failures"].as_array().is_some_and(|f| f.iter().any(|x| x["index"].as_str() == Some(i.as_str())));
            if failed {
                if partial {
                    continue;
                }
                return Err(EsError::new(500, "snapshot_restore_exception", format!("[{repo}:{snap}/{}] index [{i}] wasn't fully snapshotted - cannot restore", meta["uuid"].as_str().unwrap_or(""))));
            }
            let target = rename(i)?;
            if let Some(existing) = node.get_index(&target) {
                if existing.is_open() {
                    return Err(EsError::new(500, "snapshot_restore_exception", format!(
                        "[{repo}:{snap}/{}] cannot restore index [{target}] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name",
                        meta["uuid"].as_str().unwrap_or("")
                    )));
                }
            }
            plan.push((i.clone(), target));
        }
        let ignore_settings: Vec<String> = crate::node::pattern_list(&body["ignore_index_settings"]);
        let include_aliases = body["include_aliases"].as_bool().unwrap_or(true);
        let snap_dir = dir.join(format!("snap-{snap}"));
        let mut restored = Vec::new();
        for (src_name, target) in plan {
            let start = crate::date::now_millis();
            let src = snap_dir.join(&src_name);
            let raw: Value = serde_json::from_slice(&std::fs::read(src.join("index_meta.json"))?)?;
            let mut settings: Map<String, Value> = raw["settings"].as_object().cloned().unwrap_or_default();
            // `ignore_index_settings` drops snapshot settings, then `index_settings` apply (`RestoreService`).
            settings.retain(|k, _| !ignore_settings.iter().any(|p| p == k || p == &format!("index.{k}") || (p.contains('*') && wildcard_match(p, k))));
            if let Some(o) = body["index_settings"].as_object() {
                let flat = crate::settings::flatten(&Value::Object(o.clone()));
                let mut changed = Vec::new();
                for (k, v) in flat {
                    let k = if k.starts_with("index.") { k } else { format!("index.{k}") };
                    changed.push(k.clone());
                    settings.insert(k, v);
                }
                // Overrides are new values (the snapshot's own are restored
                // as they were).
                let merged: crate::settings::FlatSettings = settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                crate::similarity::check_changed_ranges(&merged, &changed)?;
            }
            let uuid = new_uuid();
            settings.insert("index.uuid".into(), json!(uuid));
            settings.insert("index.provided_name".into(), json!(target));
            let aliases = if include_aliases { raw["aliases"].clone() } else { json!({}) };
            if node.get_index(&target).is_some() {
                node.delete_index(&target)?;
            }
            let new_dir = node.data.join("indices").join(&uuid);
            // Every shard directory of the snapshotted index.
            let shard_dirs: Vec<String> = std::fs::read_dir(&src)?
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("shard-"))
                .collect();
            let res = (|| -> Result<Arc<IndexService>> {
                for d in &shard_dirs {
                    crate::ilm::resize::copy_tree(&src.join(d), &new_dir.join(d))?;
                }
                let m = json!({
                    "name": target, "uuid": uuid, "creation_date": raw["creation_date"],
                    "settings": settings, "mappings": raw["mappings"], "aliases": aliases, "open": true,
                });
                std::fs::write(new_dir.join("index_meta.json"), serde_json::to_vec_pretty(&m)?)?;
                let svc = Arc::new(IndexService::open(&new_dir)?);
                node.register_index(svc.clone())?;
                Ok(svc)
            })();
            let svc = match res {
                Ok(svc) => svc,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&new_dir);
                    return Err(e);
                }
            };
            let bytes: u64 = shard_dirs.iter().map(|d| dir_files(&src.join(d)).1).sum();
            // Files a Lucene shard recovers: `segments_N`, plus one segment
            // holding the documents — compound (`.si` `.cfe` `.cfs`), or for
            // a `source` repository the stored-fields-only segment
            // (`.si` `.fnm` `.fdt` `.fdx` `.fdm`).
            let source_only = settings.get("index.source_only").is_some_and(|v| v == &json!("true") || v == &json!(true));
            let nfiles = lucene_recovered_files(svc.doc_count().0 as u64, source_only);
            self.restored.lock().insert(target.clone(), (start, crate::date::now_millis(), bytes.max(1), nfiles));
            restored.push(target);
        }
        // `include_global_state`: persistent settings are replaced, legacy
        // templates put, composable / component templates replaced (`RestoreService`).
        if body["include_global_state"].as_bool().unwrap_or(false) {
            if let Some(gs) = meta.get("global_state").filter(|g| g.is_object()) {
                let obj = |k: &str| -> std::collections::BTreeMap<String, Value> { gs[k].as_object().map(|o| o.clone().into_iter().collect()).unwrap_or_default() };
                if let Some(ps) = gs["persistent_settings"].as_object() {
                    node.cluster_settings.write().0 = ps.clone();
                    crate::memory::configure(node);
                    crate::script::rate_limit::configure(node)?;
                }
                {
                    let mut t = node.templates.write();
                    for (k, v) in obj("legacy_templates") {
                        t.legacy.insert(k, v);
                    }
                    if gs.get("index_templates").is_some() {
                        t.index = obj("index_templates");
                    }
                    if gs.get("component_templates").is_some() {
                        t.component = obj("component_templates");
                    }
                }
                node.persist_state()?;
            }
        }
        let n = restored.len();
        Ok(json!({"snapshot": {"snapshot": snap, "indices": restored, "shards": {"total": n, "failed": 0, "successful": n}}}))
    }

    // ---- searchable snapshots ----------------------------------------

    /// `POST /_snapshot/{repo}/{snap}/_mount` (`TransportMountSearchableSnapshotAction`):
    /// restore one index of the snapshot under the mounted name with the
    /// searchable-snapshot settings (read-only, `index.store.type: snapshot`).
    pub fn mount(&self, node: &Node, repo: &str, snap: &str, index: &str, mounted: &str, index_settings: &Map<String, Value>, ignore: &[String], partial: bool) -> Result<Value> {
        let dir = self.dir_of(repo)?;
        let repo_uuid = self.repos.read().get(repo).and_then(|r| r["uuid"].as_str().map(str::to_string)).unwrap_or_default();
        let names = Self::list_names(&dir);
        let index_ids = |m: &Value| m["indices"].as_array().is_some_and(|a| a.iter().any(|v| v.as_str() == Some(index)));
        let metas: Vec<Value> = names.iter().filter_map(|n| Self::read_meta(&dir, repo, n).ok()).collect();
        if !metas.iter().any(index_ids) {
            return Err(EsError::index_not_found(&format!("index [{index}] not found in repository [{repo}]")));
        }
        let meta = metas.iter().find(|m| m["snapshot"].as_str() == Some(snap)).ok_or_else(|| EsError::new(500, "exception", format!("snapshot [{snap}] not found in repository [{repo}]")))?;
        let raw: Value = serde_json::from_slice(&std::fs::read(dir.join(format!("snap-{snap}")).join(index).join("index_meta.json")).map_err(|_| EsError::index_not_found(&format!("index [{index}] not found in repository [{repo}]")))?)?;
        let src_settings = raw["settings"].as_object().cloned().unwrap_or_default();
        if src_settings.get("index.store.type").and_then(Value::as_str) == Some("snapshot") {
            let g = |k: &str| src_settings.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            return Err(EsError::illegal_argument(format!(
                "index [{index}] in snapshot [{repo}/{repo_uuid}:{snap}] is a snapshot of a searchable snapshot index backed by index [{}] in snapshot [{}/{}:{}] and cannot be mounted; did you mean to restore it instead?",
                g("index.store.snapshot.index_name"),
                g("index.store.snapshot.repository_name"),
                g("index.store.snapshot.repository_uuid"),
                g("index.store.snapshot.snapshot_name")
            )));
        }
        let mut ignore: Vec<String> = ignore.to_vec();
        ignore.push("index.data_path".into());
        for k in src_settings.keys() {
            if k.starts_with("index.routing.allocation.require.") || k.starts_with("index.routing.allocation.include.") || k.starts_with("index.routing.allocation.exclude.") {
                ignore.push(k.clone());
            }
        }
        let mut settings = Map::new();
        settings.insert("index.number_of_replicas".into(), json!("0"));
        settings.insert("index.auto_expand_replicas".into(), json!("false"));
        settings.insert("index.shard.check_on_startup".into(), json!("false"));
        settings.insert("index.routing.allocation.include._tier_preference".into(), json!(if partial { "data_frozen" } else { "data_cold,data_warm,data_hot" }));
        for (k, v) in crate::settings::flatten(&Value::Object(index_settings.clone())) {
            let k = if k.starts_with("index.") { k } else { format!("index.{k}") };
            settings.insert(k, v);
        }
        if !repo_uuid.is_empty() {
            settings.insert("index.store.snapshot.repository_uuid".into(), json!(repo_uuid));
        }
        settings.insert("index.store.snapshot.repository_name".into(), json!(repo));
        settings.insert("index.store.snapshot.snapshot_name".into(), json!(snap));
        settings.insert("index.store.snapshot.snapshot_uuid".into(), meta["uuid"].clone());
        settings.insert("index.store.snapshot.index_name".into(), json!(index));
        let index_id = src_settings.get("index.uuid").cloned().unwrap_or_else(|| json!(index));
        settings.insert("index.store.snapshot.index_uuid".into(), index_id);
        settings.insert("index.store.type".into(), json!("snapshot"));
        settings.insert("index.blocks.write".into(), json!("true"));
        settings.insert("index.allocation.existing_shards_allocator".into(), json!("searchable_snapshot_allocator"));
        settings.insert("index.recovery.type".into(), json!("snapshot_prewarm"));
        if partial {
            settings.insert("index.store.snapshot.partial".into(), json!("true"));
            settings.insert("index.routing.allocation.disk.watermark.ignore".into(), json!("true"));
            settings.insert("index.shard_limit.group".into(), json!("frozen"));
        }
        let body = json!({
            "indices": [index],
            "rename_pattern": ".+",
            "rename_replacement": mounted.replace('$', "$$"),
            "index_settings": settings,
            "ignore_index_settings": ignore,
            "include_global_state": false,
            "include_aliases": false,
        });
        self.restore(node, repo, snap, &body)
    }

    /// Searchable-snapshot indices backed by repository `repo` (by uuid, else name).
    pub fn indices_using_repository(&self, node: &Node, repo: &str) -> Vec<Arc<IndexService>> {
        let uuid = self.repos.read().get(repo).and_then(|r| r["uuid"].as_str().map(str::to_string));
        node.all_indices()
            .into_iter()
            .filter(|svc| is_searchable_snapshot(svc))
            .filter(|svc| match svc.setting("index.store.snapshot.repository_uuid").filter(|u| !u.is_empty()) {
                Some(u) => uuid.as_deref() == Some(u.as_str()),
                None => svc.setting("index.store.snapshot.repository_name").as_deref() == Some(repo),
            })
            .collect()
    }

    /// Recovery details of an index restored from a snapshot in this process.
    pub fn restored_info(&self, index: &str) -> Option<(i64, i64, u64, u64)> {
        self.restored.lock().get(index).copied()
    }

    pub fn forget_restored(&self, index: &str) {
        self.restored.lock().remove(index);
    }
}

/// Does the index keep its complete `_source` (stored, unfiltered)?
fn has_complete_source(svc: &IndexService) -> bool {
    let meta = svc.meta();
    let mode = crate::synthetic::source_mode(crate::settings::get_str(&meta.settings, "index.mapping.source.mode"), &meta.mappings, crate::settings::get_str(&meta.settings, "index.mode"));
    let filtered = meta.mappings.source.as_ref().is_some_and(|s| {
        let non_empty = |k: &str| s.get(k).and_then(Value::as_array).is_some_and(|a| !a.is_empty());
        non_empty("includes") || non_empty("excludes")
    });
    mode == crate::synthetic::SourceMode::Stored && !filtered
}

/// The index metadata a `source` repository stores: read-only and marked
/// `index.source_only` (`SourceOnlySnapshotRepository.metadataToSnapshot`).
fn mark_source_only(dir: &Path) -> Result<()> {
    let p = dir.join("index_meta.json");
    let mut m: Value = serde_json::from_slice(&std::fs::read(&p)?)?;
    if let Some(st) = m["settings"].as_object_mut() {
        st.insert("index.source_only".into(), json!("true"));
        st.insert("index.blocks.write".into(), json!("true"));
    }
    std::fs::write(&p, serde_json::to_vec_pretty(&m)?)?;
    Ok(())
}

/// Is `svc` a searchable snapshot (mounted) index?
pub fn is_searchable_snapshot(svc: &IndexService) -> bool {
    svc.setting("index.store.type").as_deref() == Some("snapshot")
}

/// Is `svc` a partially cached (`shared_cache`) searchable snapshot index?
pub fn is_partial_snapshot(svc: &IndexService) -> bool {
    is_searchable_snapshot(svc) && svc.setting("index.store.snapshot.partial").as_deref() == Some("true")
}

fn copy_index(svc: &IndexService, dst: &Path) -> Result<()> {
    svc.flush()?;
    svc.persist()?;
    std::fs::create_dir_all(dst)?;
    std::fs::copy(svc.path().join("index_meta.json"), dst.join("index_meta.json"))?;
    for sh in svc.local_shards() {
        // Pin the files so no commit swaps them mid-copy (writes and searches
        // continue; the copy's commit point is the flush above).
        let _files = sh.shard.files_pinned();
        let name = format!("shard-{}", sh.id);
        crate::ilm::resize::copy_tree(&svc.path().join(&name), &dst.join(&name))?;
    }
    Ok(())
}

#[cfg(test)]
mod registration_tests {
    use super::*;

    #[test]
    fn cluster_registration_validation_does_not_open_standalone_repository() {
        let service = SnapshotService::default();
        service.restore_cluster_registrations(BTreeMap::from([
            ("writable".into(), json!({"type":"fs", "settings":{"location":"/cluster/repo"}})),
            ("readonly".into(), json!({"type":"fs", "settings":{"readonly":"true"}})),
        ]));
        assert_eq!(service.registered_readonly("writable"), Some(false));
        // Every node lists every published repository (node stats).
        assert_eq!(service.repository_names(), vec!["readonly".to_string(), "writable".to_string()]);
        assert_eq!(service.registered_readonly("readonly"), Some(true));
        assert_eq!(service.registered_readonly("missing"), None);
        assert!(service.repository_dir("writable").is_err());
        service.restore_cluster_registrations(BTreeMap::new());
        assert_eq!(service.registered_readonly("writable"), None);
    }
}

/// Files a Lucene shard's snapshot recovery reports (`_recovery`
/// `index.files`): `segments_N`, plus one segment holding the documents —
/// compound (`.si` `.cfe` `.cfs`), or for a `source` repository the
/// stored-fields-only segment (`.si` `.fnm` `.fdt` `.fdx` `.fdm`). Our store
/// layout differs, so single-node and cluster restores both report this.
pub fn lucene_recovered_files(docs: u64, source_only: bool) -> u64 {
    1 + if docs == 0 { 0 } else if source_only { 5 } else { 3 }
}

#[cfg(test)]
mod lucene_files_tests {
    #[test]
    fn counts_commit_plus_one_segment() {
        assert_eq!(super::lucene_recovered_files(0, false), 1);
        assert_eq!(super::lucene_recovered_files(1, false), 4);
        assert_eq!(super::lucene_recovered_files(1, true), 6);
    }
}
