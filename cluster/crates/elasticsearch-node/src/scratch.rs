//! The master service's scratch engine: where index-level metadata requests
//! (create / delete index, settings, aliases, templates, data streams,
//! open / close, auto-create) execute before their result is committed.
//!
//! ES computes a new cluster state on a copy (`MetadataCreateIndexService`
//! even builds a temporary `IndexService` to validate a new index) and only
//! the cluster applier changes the node's indices. Here the engine is the
//! implementation of every metadata API, so the master runs them on a copy
//! of the engine's metadata: a sparse (metadata-only) engine under
//! `<path.data>/_cluster/scratch-engine`, re-synced to the master's metadata
//! before each batch, sharing the live security service (requests are
//! authorized as on the live engine) and holding the license, lifecycle
//! policies, pipelines, synonym sets and inference endpoints requests
//! validate against. The batch publishes what the copy changed; the live
//! engines (the master's included) change only when the applier applies
//! the committed state. Nothing a request does there is visible to a
//! reader, and a delete never destroys data before its commit.
//!
//! Cost: a second metadata copy of every index on the master (parsed
//! settings and mappings, no shard data), built once when a node first
//! acts as master (O(indices) once, its writes not fsynced). After that a
//! batch is proportional to what changed: [`ScratchEngine::sync`] applies
//! only indices whose metadata version differs from what the copy holds
//! (in place), and [`ScratchEngine::changes`] extracts only the indices a
//! batch's requests touched.

use crate::mirror::{self, ApplyReport, UuidMap};
use elasticsearch_cluster::state::{IndexMetadata, Metadata};
use elasticsearch_engine::node::Node as Engine;
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// What the copy holds, so a sync touches only what changed.
#[derive(Default)]
struct Synced {
    /// Index name → (cluster uuid, metadata version) the copy matches.
    indices: HashMap<String, (String, u64)>,
    /// Copies a batch changed without a commit: re-created at the next sync.
    dirty: HashSet<String>,
    /// The globals and validation customs last applied (`None`: re-apply).
    globals: Option<Metadata>,
}

/// One index copy as a batch sees it before and after its requests: the
/// service instance, its field lookup (replaced on any mapping change),
/// its settings / aliases fingerprint and open state.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    svc: usize,
    lookup: usize,
    meta: u64,
    open: bool,
    /// The lifecycle execution state's change marker (O(1)).
    lifecycle: u64,
}

/// An engine's indices and globals at one point of a batch.
pub struct Snapshot {
    indices: HashMap<String, Shape>,
    globals: Metadata,
    /// The customs captured: `None` none, `Some(None)` all, `Some(Some(k))`
    /// only family `k`.
    customs: Option<Option<&'static str>>,
    /// Publications of a kept job from this snapshot so far (retries).
    pub attempt: u8,
}

/// `engine` as it is now (cheap: no index metadata is serialized). With
/// `customs`, the engine's extra metadata (pipelines, lifecycle policies,
/// scripts, ...) is captured too (kept jobs change those).
pub fn snapshot_of(engine: &Engine, customs: bool) -> Snapshot {
    snapshot_scoped(engine, customs.then_some(None))
}

/// [`snapshot_of`] with the customs it captures (see `Snapshot::customs`).
pub fn snapshot_scoped(engine: &Engine, customs: Option<Option<&'static str>>) -> Snapshot {
    let indices = engine.all_indices().into_iter().map(|svc| {
        let lifecycle = elasticsearch_engine::ilm::index_lifecycle_generation(engine, &svc);
        let shape = Shape { svc: Arc::as_ptr(&svc) as usize, lookup: Arc::as_ptr(&svc.lookup()) as usize, meta: svc.meta_fingerprint(), open: svc.is_open(), lifecycle };
        (svc.name.clone(), shape)
    }).collect();
    let mut globals = Metadata::default();
    mirror::extract_globals(engine, &mut globals);
    if let Some(only) = customs {
        crate::meta_extra::extract_only(engine, &mut globals, only);
    }
    Snapshot { indices, globals, customs, attempt: 0 }
}

/// What changed on `engine` between `before` and now, as metadata against
/// `base`: only the touched indices are extracted, an extraction equal to
/// `base`'s entry is no change. Returns the per-index changes and the
/// snapshot after (its globals included).
pub fn changes_of(engine: &Engine, uuids: &mut UuidMap, before: &Snapshot, base: &Metadata, customs: bool) -> (BTreeMap<String, Option<IndexMetadata>>, Snapshot) {
    let after = snapshot_of(engine, customs);
    let out = changes_between(engine, uuids, &[before], &after, base);
    (out, after)
}

/// The indices that changed on `engine` between any of `befores` and
/// `after` (a snapshot of it now), extracted once, as metadata against
/// `base`; an extraction equal to `base`'s entry is no change.
pub fn changes_between(engine: &Engine, uuids: &mut UuidMap, befores: &[&Snapshot], after: &Snapshot, base: &Metadata) -> BTreeMap<String, Option<IndexMetadata>> {
    let mut touched: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    for before in befores {
        touched.extend(after.indices.iter().filter(|(name, shape)| before.indices.get(*name) != Some(*shape)).map(|(name, _)| name));
        touched.extend(before.indices.keys().filter(|name| !after.indices.contains_key(*name)));
    }
    let mut out = BTreeMap::new();
    for name in touched {
        match engine.get_index(name).filter(|_| after.indices.contains_key(name)) {
            Some(svc) => {
                let im = mirror::extract_index(engine, &svc, base, uuids);
                if base.indices.get(name) != Some(&im) {
                    out.insert(name.clone(), Some(im));
                }
            }
            None => {
                if base.indices.contains_key(name) {
                    out.insert(name.clone(), None);
                }
            }
        }
    }
    out
}

impl Snapshot {
    pub fn globals(&self) -> &Metadata {
        &self.globals
    }

    /// The customs it captured (see the field).
    pub fn customs(&self) -> Option<Option<&'static str>> {
        self.customs
    }

    /// The indices the engine held.
    pub fn index_names(&self) -> impl Iterator<Item = &String> + '_ {
        self.indices.keys()
    }
}

/// What the successful jobs of a batch so far changed on the copy, over
/// the batch's base: a failed job is undone to this, not to the base (an
/// earlier job's acknowledged change to the same index or globals stays).
#[derive(Default)]
pub struct Overlay {
    indices: BTreeMap<String, Option<IndexMetadata>>,
    globals: Option<Metadata>,
}

pub struct ScratchEngine {
    pub engine: Arc<Engine>,
    /// The node's engine 0: an index state without the lifecycle state (a
    /// master from before it was published) takes engine 0's.
    live: std::sync::Weak<Engine>,
    pub router: axum::Router,
    pub uuids: Mutex<UuidMap>,
    synced: Mutex<Synced>,
}

/// The customs the copy holds for validation (see `meta_extra::apply_for_validation`).
const VALIDATION_CUSTOMS: &[&str] = &[
    "repositories",
    crate::meta_extra::LICENSE_CUSTOM,
    crate::meta_extra::ILM_CUSTOM,
    crate::meta_extra::INGEST_CUSTOM,
    crate::meta_extra::SYNONYMS_CUSTOM,
    crate::meta_extra::RULESETS_CUSTOM,
    crate::meta_extra::SEARCH_APPS_CUSTOM,
    crate::meta_extra::INFERENCE_CUSTOM,
    crate::meta_extra::ESQL_VIEWS_CUSTOM,
];

impl ScratchEngine {
    /// A fresh scratch engine for `live` (the node's engine 0).
    pub fn open(data_path: &std::path::Path, live: &Arc<Engine>) -> Result<ScratchEngine, String> {
        let dir = data_path.join("_cluster").join("scratch-engine");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        // Rebuilt from the cluster state at every open: nothing in it needs
        // to survive a crash, so its writes skip fsync.
        elasticsearch_engine::hooks::set_volatile_root(&dir);
        let engine = Engine::open_with_security_named(&dir.join("engine"), live.security.clone(), &live.name).map_err(|e| e.to_string())?;
        *engine.node_settings.write() = live.node_settings.read().clone();
        *engine.transport_address.write() = live.transport_address.read().clone();
        *engine.cluster_transport_addresses.write() = live.cluster_transport_addresses.read().clone();
        // A copy runs no lifecycle (rollovers, deletes) of its own; nothing
        // else starts background work for an engine the cluster opens (the
        // HTTP layer's `serve_listener` would, and is never used for it).
        engine.ilm.drive_externally();
        // License-dependent index creation (synthetic source falling back to
        // stored under a basic license) follows the license the copy holds,
        // as on any node that serves requests.
        elasticsearch_engine::license::start(&engine);
        let router = elasticsearch_http::router(engine.clone());
        let uuids = Mutex::new(UuidMap::load(&dir.join("uuids.json")));
        Ok(ScratchEngine { engine, router, uuids, synced: Mutex::new(Synced::default()), live: Arc::downgrade(live) })
    }

    /// Make the copy hold `md` (the committed metadata at the start of a
    /// batch). Only indices whose metadata version (or uuid) differs from
    /// what the copy holds are applied, in place; globals and validation
    /// resources only when they changed. The first sync (and the first
    /// after an open, e.g. on a new master) builds every index once.
    pub fn sync(&self, md: &Metadata) {
        let mut synced = self.synced.lock();
        let mut uuids = self.uuids.lock();
        let mut rep = ApplyReport::default();
        let validation_changed = synced.globals.as_ref().is_none_or(|g| VALIDATION_CUSTOMS.iter().any(|k| g.customs.get(*k) != md.customs.get(*k)));
        if validation_changed {
            for (what, e) in crate::meta_extra::apply_for_validation(&self.engine, md) {
                tracing::debug!(%what, "scratch engine: {e}");
            }
            if let Some(e) = crate::meta_extra::apply_inference(&self.engine, md) {
                tracing::debug!("scratch engine: inference endpoints: {e}");
            }
        }
        // Copies a failed batch changed, and copies of indices gone or
        // re-created since: removed (mappings only grow, so a copy that
        // holds more than `md` is re-created rather than updated).
        let stale: Vec<String> = synced.indices.iter()
            .filter(|(name, (uuid, _))| synced.dirty.contains(*name) || md.indices.get(*name).is_none_or(|im| &im.uuid != uuid))
            .map(|(name, _)| name.clone())
            .chain(synced.dirty.iter().cloned())
            .collect::<HashSet<String>>().into_iter().collect();
        for name in stale {
            if self.engine.get_index(&name).is_some() {
                if let Err(e) = crate::meta_extra::delete_local_copy(&self.engine, &name) {
                    tracing::debug!(index = %name, "scratch engine: {e}");
                }
            }
            uuids.indices.remove(&name);
            synced.indices.remove(&name);
        }
        synced.dirty.clear();
        for (name, im) in &md.indices {
            if synced.indices.get(name).is_some_and(|(uuid, version)| uuid == &im.uuid && *version == im.version) {
                continue;
            }
            let errors = rep.errors.len();
            mirror::apply_index(&self.engine, name, im, &mut uuids, &mut rep);
            if im.lifecycle.is_none() {
                self.seed_lifecycle_from_live(name);
            }
            if rep.errors.len() == errors {
                synced.indices.insert(name.clone(), (im.uuid.clone(), im.version));
            } else {
                // Not synced: a copy holding an older (or partial) form
                // would let a request extract it, and publish a revert.
                // Without a copy the request fails (index not found); the
                // next sync tries again.
                if self.engine.get_index(name).is_some() {
                    if let Err(e) = crate::meta_extra::delete_local_copy(&self.engine, name) {
                        tracing::debug!(index = %name, "scratch engine: {e}");
                    }
                }
                uuids.indices.remove(name);
            }
        }
        let globals_changed = synced.globals.as_ref().is_none_or(|g| mirror::globals_differ(g, md));
        if globals_changed {
            mirror::apply_globals(&self.engine, md);
        }
        if globals_changed || validation_changed {
            synced.globals = Some(globals_of(md));
        }
        for (index, e) in &rep.errors {
            tracing::warn!(%index, "scratch engine: failed to mirror metadata: {e}");
        }
        let _ = uuids.save();
    }

    /// Take engine 0's lifecycle state for `name` (see `live`).
    fn seed_lifecycle_from_live(&self, name: &str) {
        let Some(live) = self.live.upgrade() else { return };
        let (Some(lsvc), Some(svc)) = (live.get_index(name), self.engine.get_index(name)) else { return };
        if elasticsearch_engine::ilm::index_lifecycle_state(&self.engine, &svc).is_some() {
            return;
        }
        if let Some(state) = elasticsearch_engine::ilm::index_lifecycle_state(&live, &lsvc) {
            if let Err(e) = elasticsearch_engine::ilm::restore_index_lifecycle_state(&self.engine, &svc, Some(&state)) {
                tracing::debug!(index = %name, "scratch engine: lifecycle state not seeded: {}", e.reason);
            }
        }
    }

    /// The copy as it is now (cheap: no metadata is serialized).
    pub fn snapshot(&self) -> Snapshot {
        snapshot_of(&self.engine, false)
    }

    /// What a batch's requests changed on the copy between `before` and now,
    /// as metadata against `base` (the committed metadata): only the
    /// touched indices are extracted, an extraction equal to `base`'s entry
    /// is no change. Returns the per-index changes and the globals after.
    pub fn changes(&self, before: &Snapshot, base: &Metadata) -> (BTreeMap<String, Option<IndexMetadata>>, Metadata) {
        let (out, after) = changes_of(&self.engine, &mut self.uuids.lock(), before, base, false);
        (out, after.globals)
    }

    /// The batch's publication committed `committed` (index changes as
    /// published) and failed for `failed` (indices the batch touched whose
    /// change was not committed as the copy holds it).
    pub fn settle(&self, committed: &BTreeMap<String, Option<IndexMetadata>>, failed: impl IntoIterator<Item = String>) {
        let mut synced = self.synced.lock();
        for (name, change) in committed {
            match change {
                Some(im) => {
                    synced.indices.insert(name.clone(), (im.uuid.clone(), im.version));
                }
                None => {
                    synced.indices.remove(name);
                }
            }
        }
        for name in failed {
            synced.dirty.insert(name);
        }
    }

    /// The indices (and whether the globals) a request changed on the copy
    /// since `before`.
    pub fn touched(&self, before: &Snapshot) -> (Vec<String>, bool) {
        let after = self.snapshot();
        let mut names: Vec<String> = after.indices.iter().filter(|(k, v)| before.indices.get(*k) != Some(*v)).map(|(k, _)| k.clone()).collect();
        names.extend(before.indices.keys().filter(|k| !after.indices.contains_key(*k)).cloned());
        (names, mirror::globals_differ(&before.globals, &after.globals))
    }

    /// Record what a successful job changed since `job_before` into the
    /// batch's `overlay` (over `base`).
    pub fn record(&self, job_before: &Snapshot, base: &Metadata, overlay: &mut Overlay) {
        let (changes, after) = changes_of(&self.engine, &mut self.uuids.lock(), job_before, base, false);
        let touched: Vec<String> = after.indices.iter().filter(|(k, v)| job_before.indices.get(*k) != Some(*v)).map(|(k, _)| k.clone())
            .chain(job_before.indices.keys().filter(|k| !after.indices.contains_key(*k)).cloned()).collect();
        for name in touched {
            match changes.get(&name) {
                Some(change) => {
                    overlay.indices.insert(name, change.clone());
                }
                // Back to the base's form (or absent from both).
                None => {
                    overlay.indices.remove(&name);
                }
            }
        }
        if mirror::globals_differ(&job_before.globals, &after.globals) {
            overlay.globals = Some(after.globals);
        }
    }

    /// Undo a failed request's partial changes on the copy: `names` (and
    /// the globals) back to `md`, the committed metadata, with the batch's
    /// earlier successful changes (`overlay`) on top.
    pub fn restore_over(&self, names: &[String], globals: bool, md: &Metadata, overlay: &Overlay) {
        let mut synced = self.synced.lock();
        let mut uuids = self.uuids.lock();
        let mut rep = ApplyReport::default();
        for name in names {
            if self.engine.get_index(name).is_some() {
                if let Err(e) = crate::meta_extra::delete_local_copy(&self.engine, name) {
                    tracing::debug!(index = %name, "scratch engine: {e}");
                }
            }
            uuids.indices.remove(name);
            synced.indices.remove(name);
            let (target, committed) = match overlay.indices.get(name) {
                Some(change) => (change.as_ref(), false),
                None => (md.indices.get(name), true),
            };
            if let Some(im) = target {
                mirror::apply_index(&self.engine, name, im, &mut uuids, &mut rep);
                if committed {
                    synced.indices.insert(name.clone(), (im.uuid.clone(), im.version));
                } else {
                    // An uncommitted form: re-synced after the batch.
                    synced.dirty.insert(name.clone());
                }
            }
        }
        if globals {
            mirror::apply_globals(&self.engine, overlay.globals.as_ref().unwrap_or(md));
            synced.globals = None;
        }
        for (index, e) in &rep.errors {
            tracing::warn!(%index, "scratch engine: failed to restore metadata: {e}");
        }
        let _ = uuids.save();
    }

    /// Undo a failed request's partial changes on the copy: `names` (and
    /// the globals) back to `md`, the committed metadata.
    pub fn restore(&self, names: &[String], globals: bool, md: &Metadata) {
        self.restore_over(names, globals, md, &Overlay::default());
    }

    /// Forget what the copy holds (after a panic in a batch): the next sync
    /// rebuilds every index and the globals.
    pub fn reset(&self) {
        let mut synced = self.synced.lock();
        let names: Vec<String> = self.engine.all_indices().iter().map(|s| s.name.clone()).collect();
        synced.dirty.extend(names);
        synced.indices.clear();
        synced.globals = None;
    }

    /// Re-apply globals at the next sync (a batch changed them on the copy
    /// without a commit).
    pub fn globals_dirty(&self) {
        self.synced.lock().globals = None;
    }

    /// Close any shard copy a request opened on the copy (a created index
    /// opens its shards): the copy holds metadata only.
    pub fn close_copies(&self, indices: impl IntoIterator<Item = String>) {
        for name in indices {
            let Some(svc) = self.engine.get_index(&name) else { continue };
            for copy in svc.local_shards() {
                if let Err(e) = svc.close_shard(copy.id) {
                    tracing::debug!(index = %name, "scratch engine: {}", e.reason);
                }
            }
        }
    }
}

/// The parts of `md` [`ScratchEngine::sync`] compares to decide re-applying.
fn globals_of(md: &Metadata) -> Metadata {
    let mut g = Metadata::default();
    g.index_templates = md.index_templates.clone();
    g.component_templates = md.component_templates.clone();
    g.legacy_templates = md.legacy_templates.clone();
    g.persistent_settings = md.persistent_settings.clone();
    g.transient_settings = md.transient_settings.clone();
    for k in VALIDATION_CUSTOMS.iter().copied().chain([mirror::DATA_STREAMS_CUSTOM]) {
        if let Some(v) = md.customs.get(k) {
            g.customs.insert(k.to_string(), v.clone());
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn committed_views_fence_create_and_auto_create_until_removed() {
        let tmp=tempfile::tempdir().unwrap();
        let live=Engine::open(&tmp.path().join("live")).unwrap();
        let scratch=ScratchEngine::open(tmp.path(),&live).unwrap();
        let mut md=Metadata::default();scratch.sync(&md);
        md.customs.insert(crate::meta_extra::ESQL_VIEWS_CUSTOM.into(),json!({"myview":"FROM source"}));
        scratch.sync(&md);
        let error=scratch.engine.create_index("myview",&json!({})).err().expect("view name conflict");
        assert_eq!(error.status,400);assert!(error.reason.contains("already exists as an ESQL view"));
        let error=scratch.engine.write_index("myview",true).err().expect("views are not auto-create targets");
        assert_eq!(error.status,404);assert_eq!(error.kind,"index_not_found_exception");
        assert!(!elasticsearch_engine::esql::views::exists(&live,"myview"),"scratch must not change live catalog");
        assert!(crate::meta_extra::apply_esql_views(&live,&md).is_none());
        assert!(elasticsearch_engine::esql::views::exists(&live,"myview"));
        md.customs.remove(crate::meta_extra::ESQL_VIEWS_CUSTOM);
        scratch.sync(&md);
        assert!(crate::meta_extra::apply_esql_views(&live,&md).is_none());
        assert!(!elasticsearch_engine::esql::views::exists(&live,"myview"),"removing last view clears follower catalog");
        scratch.engine.create_index("myview",&json!({})).unwrap();
    }

    #[tokio::test]
    async fn default_repository_validation_tracks_committed_catalog_without_opening_storage() {
        use axum::{body::{Body, to_bytes}, http::Request};
        use tower::ServiceExt;

        async fn set_default(scratch: &ScratchEngine, value: serde_json::Value) -> (u16, serde_json::Value) {
            let response = scratch.router.clone().oneshot(Request::builder()
                .method("PUT").uri("/_cluster/settings").header("content-type", "application/json")
                .body(Body::from(json!({"persistent": {"repositories.default_repository": value}}).to_string())).unwrap())
                .await.unwrap();
            let status = response.status().as_u16();
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            (status, serde_json::from_slice(&body).unwrap())
        }

        let tmp = tempfile::tempdir().unwrap();
        let live = Engine::open(&tmp.path().join("live")).unwrap();
        let scratch = ScratchEngine::open(tmp.path(), &live).unwrap();
        let mut md = Metadata::default();
        scratch.sync(&md);
        assert_eq!(set_default(&scratch, json!("repo")).await.0, 400);
        md.customs.insert("repositories".into(), json!({"repo": {
            "type": "fs", "settings": {"location": tmp.path().join("must-not-be-opened")}
        }}));
        scratch.sync(&md);
        let response = set_default(&scratch, json!("repo")).await;
        assert_eq!(response.0, 200, "{response:?}");
        assert!(!scratch.engine.snapshots.has_repository("repo"), "validation must not register repository storage");
        assert!(!tmp.path().join("must-not-be-opened").exists());
        assert_eq!(live.snapshots.registered_readonly("repo"), None, "scratch sync cannot change live registration");
        assert_eq!(set_default(&scratch, serde_json::Value::Null).await.0, 200);

        md.customs.insert("repositories".into(), json!({"repo": {"type": "fs", "settings": {"readonly": true}}}));
        scratch.sync(&md);
        let response = set_default(&scratch, json!("repo")).await;
        assert_eq!(response.0, 400, "{response:?}");
        assert!(response.1.to_string().contains("marked as read-only"), "{response:?}");
        md.customs.remove("repositories");
        scratch.sync(&md);
        let response = set_default(&scratch, json!("repo")).await;
        assert_eq!(response.0, 400, "{response:?}");
        assert!(response.1.to_string().contains("is not registered"), "{response:?}");
    }

    /// A rich fixture straight from an engine: explicit and derived
    /// settings, `_meta`, dynamic templates, runtime fields, aliases with
    /// filter / routing / hidden flags, a hidden index, routing shards.
    fn fixture(live: &Engine) -> Metadata {
        live.create_index("rich", &json!({
            "settings": {"number_of_shards": 2, "number_of_routing_shards": 8, "number_of_replicas": 1, "index.hidden": true,
                "refresh_interval": "7s", "analysis": {"analyzer": {"my": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]}}}},
            "mappings": {"_meta": {"owner": "t"}, "dynamic_templates": [{"strings": {"match_mapping_type": "string", "mapping": {"type": "keyword"}}}],
                "runtime": {"r": {"type": "long"}}, "properties": {"t": {"type": "text", "analyzer": "my"}, "n": {"type": "long"}, "o": {"properties": {"k": {"type": "keyword"}}}}},
            "aliases": {"a1": {"filter": {"term": {"n": 1}}, "routing": "x"}, "a2": {"is_hidden": true, "is_write_index": true}}
        })).unwrap();
        live.create_index("plain", &json!({})).unwrap();
        mirror::extract(live, &Metadata::default(), &mut UuidMap::default())
    }

    #[test]
    fn a_synced_copy_round_trips_the_metadata_exactly_and_reports_no_change() {
        let tmp = tempfile::tempdir().unwrap();
        let live = Engine::open(&tmp.path().join("live")).unwrap();
        let md = fixture(&live);
        let scratch = ScratchEngine::open(tmp.path(), &live).unwrap();
        scratch.sync(&md);
        let mut uuids = scratch.uuids.lock();
        for (name, im) in &md.indices {
            let svc = scratch.engine.get_index(name).expect("copy exists");
            let back = mirror::extract_index(&scratch.engine, &svc, &md, &mut uuids);
            assert_eq!(&back, im, "index [{name}] round-trips exactly");
        }
        drop(uuids);
        let mut globals = Metadata::default();
        mirror::extract_globals(&scratch.engine, &mut globals);
        assert!(!mirror::globals_differ(&globals, &md), "templates, settings, data streams round-trip");
        // Nothing ran: no change; a second sync applies nothing.
        let before = scratch.snapshot();
        scratch.sync(&md);
        let (changes, after) = scratch.changes(&before, &md);
        assert!(changes.is_empty(), "{changes:?}");
        assert!(!mirror::globals_differ(&after, &md));
    }

    #[test]
    fn a_request_on_the_copy_changes_only_the_touched_index() {
        let tmp = tempfile::tempdir().unwrap();
        let live = Engine::open(&tmp.path().join("live")).unwrap();
        let md = fixture(&live);
        let scratch = ScratchEngine::open(tmp.path(), &live).unwrap();
        scratch.sync(&md);
        let before = scratch.snapshot();
        scratch.engine.get_index("plain").unwrap().put_mapping(&json!({"properties": {"added": {"type": "keyword"}}})).unwrap();
        scratch.engine.create_index("fresh", &json!({})).unwrap();
        let (changes, _) = scratch.changes(&before, &md);
        assert_eq!(changes.keys().collect::<Vec<_>>(), vec!["fresh", "plain"], "only touched indices are extracted");
        let plain = changes["plain"].as_ref().unwrap();
        assert_eq!(plain.mapping_version, md.indices["plain"].mapping_version + 1);
        assert!(plain.settings.ptr_eq(&md.indices["plain"].settings), "unchanged parts are the committed values");
        assert!(live.get_index("plain").unwrap().mapping_json()["properties"].get("added").is_none(), "the live engine is untouched");
        // Committed: the copy is in step without being rebuilt.
        let mut next = md.clone();
        for (k, v) in &changes {
            next.indices.insert(k.clone(), v.clone().unwrap());
        }
        scratch.settle(&changes, Vec::new());
        let plain_svc = Arc::as_ptr(&scratch.engine.get_index("plain").unwrap()) as usize;
        scratch.sync(&next);
        assert_eq!(Arc::as_ptr(&scratch.engine.get_index("plain").unwrap()) as usize, plain_svc, "no re-creation after a commit");
    }

    /// A failed request in a batch is undone to the state after
    /// the batch's earlier successful requests, not to the base: their
    /// acknowledged changes to the same index and to the globals stay.
    #[test]
    fn a_failed_request_keeps_earlier_changes_of_its_batch() {
        let tmp = tempfile::tempdir().unwrap();
        let live = Engine::open(&tmp.path().join("live")).unwrap();
        let md = fixture(&live);
        let scratch = ScratchEngine::open(tmp.path(), &live).unwrap();
        scratch.sync(&md);
        let batch_before = scratch.snapshot();
        let mut overlay = Overlay::default();
        // Job 1 (acknowledged): a field on `plain` and a cluster setting.
        let job1 = scratch.snapshot();
        scratch.engine.get_index("plain").unwrap().put_mapping(&json!({"properties": {"first": {"type": "keyword"}}})).unwrap();
        scratch.engine.cluster_settings.write().0.insert("search.default_search_timeout".into(), json!("5s"));
        scratch.record(&job1, &md, &mut overlay);
        // Job 2 (fails part-way): another field on `plain`, another setting.
        let job2 = scratch.snapshot();
        scratch.engine.get_index("plain").unwrap().put_mapping(&json!({"properties": {"second": {"type": "keyword"}}})).unwrap();
        scratch.engine.cluster_settings.write().0.insert("action.auto_create_index".into(), json!("false"));
        let (names, globals) = scratch.touched(&job2);
        assert_eq!(names, vec!["plain".to_string()]);
        assert!(globals);
        scratch.restore_over(&names, globals, &md, &overlay);
        let mapping = scratch.engine.get_index("plain").unwrap().mapping_json();
        assert!(mapping["properties"].get("first").is_some(), "job 1's field survives job 2's undo");
        assert!(mapping["properties"].get("second").is_none(), "job 2's field is undone");
        let (persistent, _) = scratch.engine.cluster_settings.read().clone();
        assert!(persistent.contains_key("search.default_search_timeout"));
        assert!(!persistent.contains_key("action.auto_create_index"));
        // The batch still publishes job 1's change.
        let (changes, after) = scratch.changes(&batch_before, &md);
        assert!(changes["plain"].as_ref().unwrap().mappings["properties"].get("first").is_some());
        assert!(after.persistent_settings.contains_key("search.default_search_timeout"));
    }
}
