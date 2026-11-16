//! Keeping each node's engine in step with the cluster metadata.
//!
//! The engine already implements every metadata API (index creation with
//! templates, mappings, settings validation, aliases, data streams, ...).
//! Rather than re-implement them, the elected master executes the REST
//! request against its own engine and then [`extract`]s the resulting
//! metadata into the cluster state it publishes; every other node
//! [`apply`]s the published metadata to its engine. Thus all nodes hold an
//! identical *metadata* mirror of every index, while shard *data* exists
//! only where the routing table put a copy.
//!
//! Index uuids: the engine mints its own uuid per created index, so each
//! node's mirror has a different local uuid. [`UuidMap`] records local →
//! cluster uuid so responses can be rewritten to the canonical value (see
//! `docs/ENGINE_HOOKS.md` #1).

use elasticsearch_cluster::state::{IndexMetadata, IndexState, Metadata};
use elasticsearch_engine::node::Node as Engine;
use elasticsearch_engine::settings::{self as es_settings, FlatSettings};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Settings the engine computes per index and that must not be copied
/// into another node's create request.
const PER_NODE_SETTINGS: &[&str] = &["index.uuid", "index.creation_date", "index.provided_name", "index.version.created"];

/// Engine data streams live in metadata customs under this key.
pub const DATA_STREAMS_CUSTOM: &str = "data_stream";

/// Cluster uuid ↔ local engine uuid, persisted next to the engine data.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UuidMap {
    /// index name → (cluster uuid, local engine uuid)
    pub indices: BTreeMap<String, (String, String)>,
    #[serde(skip)]
    path: Option<PathBuf>,
    /// Deterministic mapping conflicts are bounded per local incarnation.
    #[serde(skip)]
    mapping_conflicts: BTreeMap<String, (PathBuf, u64, u8)>,
}

impl UuidMap {
    /// Load the map at `path`. A missing file is an empty map (a new node);
    /// any other read failure, or content that does not parse, is an error:
    /// an unreadable map is never mistaken for an empty one.
    pub fn load(path: &Path) -> std::io::Result<UuidMap> {
        let mut m: UuidMap = match std::fs::read(path) {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("index uuid map [{}] is corrupt: {e}", path.display())))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => UuidMap::default(),
            Err(e) => return Err(std::io::Error::new(e.kind(), format!("cannot read index uuid map [{}]: {e}", path.display()))),
        };
        m.path = Some(path.to_path_buf());
        Ok(m)
    }

    /// [`Self::load`] at node start. A map that cannot be read fails the
    /// start. A corrupt one (torn or garbage content) is moved aside, kept
    /// for inspection, and replaced by an empty map: the map is only a
    /// cache of identities each copy also records itself (its metadata
    /// file's cluster uuid, its shard copy record), and the metadata
    /// application reconstructs them from there ([`incarnation`]); a copy
    /// whose identity cannot be established is quarantined, never deleted.
    pub fn load_at_start(path: &Path) -> std::io::Result<UuidMap> {
        match Self::load(path) {
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                let aside = path.with_extension(format!("corrupt-{}", elasticsearch_engine::date::now_millis()));
                std::fs::rename(path, &aside)?;
                if let Some(dir) = path.parent() {
                    elasticsearch_engine::rustic_core::platform::sync_dir(dir)?;
                }
                tracing::error!(map = %path.display(), kept = %aside.display(), "{e}; identities are reconstructed from the local copies");
                let mut m = UuidMap::default();
                m.path = Some(path.to_path_buf());
                Ok(m)
            }
            r => r,
        }
    }

    /// Persist durably: temporary file, fsync, rename, fsync of the
    /// directory. Any failure is returned (the previous file stays).
    pub(crate) fn save(&self) -> std::io::Result<()> {
        let Some(p) = &self.path else { return Ok(()) };
        save_test_hook::check(p)?;
        let tmp = p.with_extension("tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&serde_json::to_vec(self)?)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, p)?;
        if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            elasticsearch_engine::rustic_core::platform::sync_dir(dir)?;
        }
        Ok(())
    }

    pub fn cluster_uuid(&self, index: &str) -> Option<&str> {
        self.indices.get(index).map(|(c, _)| c.as_str())
    }

    /// Rewrite every local uuid occurring in `body` to its cluster uuid.
    /// Uuids are 22 random base64 chars, so plain substring replacement is
    /// collision-free.
    pub fn canonicalize<'b>(&self, body: &'b str) -> std::borrow::Cow<'b, str> {
        canonicalize_with(&self.renames(), body)
    }

    /// `(local, cluster)` uuid pairs that differ: what [`Self::canonicalize`]
    /// rewrites. Taken under the map's lock, so the (possibly megabytes
    /// long) response is scanned without it.
    pub fn renames(&self) -> Vec<(String, String)> {
        self.indices.values().filter(|(c, l)| c != l).map(|(c, l)| (l.clone(), c.clone())).collect()
    }
}

/// Rewrite every `(local, cluster)` uuid of `renames` occurring in `body`;
/// borrowed (no copy of the body) when none occurs, as for search hits.
pub fn canonicalize_with<'b>(renames: &[(String, String)], body: &'b str) -> std::borrow::Cow<'b, str> {
    let mut out = std::borrow::Cow::Borrowed(body);
    for (local, cluster) in renames {
        if out.contains(local.as_str()) {
            out = std::borrow::Cow::Owned(out.replace(local.as_str(), cluster));
        }
    }
    out
}

fn index_settings(svc: &elasticsearch_engine::index::IndexService) -> Map<String, Value> {
    match es_settings::as_flat_json(&svc.meta().settings) {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

fn parse_u32(v: Option<&Value>, default: u32) -> u32 {
    v.and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|n| n.to_string()))).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Tests: make [`UuidMap::save`] fail for maps under a directory.
pub(crate) mod save_test_hook {
    #[cfg(test)]
    static FAIL_UNDER: parking_lot::Mutex<Vec<std::path::PathBuf>> = parking_lot::Mutex::new(Vec::new());
    #[cfg(test)]
    pub fn fail_under(dir: &std::path::Path, fail: bool) {
        let mut v = FAIL_UNDER.lock();
        v.retain(|d| d != dir);
        if fail { v.push(dir.to_path_buf()); }
    }
    #[allow(unused_variables)]
    pub fn check(path: &std::path::Path) -> std::io::Result<()> {
        #[cfg(test)]
        if FAIL_UNDER.lock().iter().any(|d| path.starts_with(d)) {
            return Err(std::io::Error::other("injected uuid map write failure"));
        }
        Ok(())
    }
}

/// Which cluster index incarnation a local engine index is, from the
/// evidence the copy itself and the uuid map hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Incarnation {
    /// The incarnation asked about.
    Same,
    /// Proven to be another incarnation (the reason names the evidence).
    Other(String),
    /// Nothing proves either (or the evidence conflicts).
    Unknown(String),
}

/// The incarnation of the local index `svc` relative to the cluster index
/// uuid `cluster_uuid`. Evidence, any of which suffices when the rest does
/// not contradict it: the cluster uuid recorded in the copy's own metadata
/// file ([`IndexService::cluster_uuid`], persisted), the uuid map's entry
/// for this exact local index (same local uuid), the shard copy record
/// (`cluster-copy.json`, which names the index uuid), and the local uuid
/// itself (the engine adopts the cluster uuid when it can). A map entry
/// for another local uuid is stale and proves nothing.
pub fn incarnation(svc: &elasticsearch_engine::index::IndexService, cluster_uuid: &str, uuids: &UuidMap) -> Incarnation {
    let mut same: Vec<&'static str> = Vec::new();
    let mut other: Vec<String> = Vec::new();
    let mut note = |c: &str, source: &'static str| {
        if c == cluster_uuid { same.push(source) } else { other.push(format!("{source} names incarnation [{c}]")) }
    };
    if let Some(c) = svc.cluster_uuid() {
        note(c, "the copy's metadata");
    }
    if let Some((c, l)) = uuids.indices.get(&svc.name) {
        if l == &svc.uuid {
            note(c, "the uuid map");
        }
    }
    if let Some(rec) = crate::shards::read_copy_record(svc.path()) {
        note(&rec.index_uuid, "the shard copy record");
    }
    if svc.uuid == cluster_uuid {
        same.push("the local uuid");
    }
    match (same.is_empty(), other.is_empty()) {
        (false, true) => Incarnation::Same,
        (true, false) => Incarnation::Other(other.join("; ")),
        (true, true) => Incarnation::Unknown(format!("nothing records which incarnation local index [{}] is", svc.uuid)),
        (false, false) => Incarnation::Unknown(format!("conflicting identity: {} name [{cluster_uuid}], but {}", same.join(", "), other.join("; "))),
    }
}

/// Whether the index directory holds any shard data (a shard store, a
/// copy record, a store set aside for reuse) rather than metadata only.
pub fn holds_shard_data(index_path: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(index_path) else { return true };
    entries.flatten().any(|e| {
        let n = e.file_name();
        let n = n.to_string_lossy();
        n.starts_with("shard-") || n == "cluster-copy.json"
    })
}

/// Make sure the local index `name` (if any) is the incarnation `im`
/// before it is applied: an index proven to be another incarnation is
/// removed (as a deleted index is), one whose identity cannot be
/// established is quarantined when it holds shard data (never deleted) and
/// removed when it is only a metadata shell, and a proven one has its
/// mapping recorded. Returns false when the name is still held by a local
/// index that is not `im` (a quarantine failed): nothing may be applied to
/// it then.
fn reconcile_incarnation(engine: &Engine, name: &str, im: &IndexMetadata, md: &Metadata, uuids: &mut UuidMap, rep: &mut ApplyReport) -> bool {
    let Some(svc) = engine.get_index(name) else { return true };
    if uuids.indices.get(name).is_some_and(|(c, l)| c == &im.uuid && l == &svc.uuid) && svc.cluster_uuid().is_none_or(|c| c == im.uuid) {
        return true;
    }
    match incarnation(&svc, &im.uuid, uuids) {
        Incarnation::Same => {
            tracing::info!(index = %name, uuid = %im.uuid, local = %svc.uuid, "local index identity reconstructed from the copy");
            uuids.indices.insert(name.to_string(), (im.uuid.clone(), svc.uuid.clone()));
            rep.reconstructed.push(name.to_string());
            true
        }
        Incarnation::Other(why) => {
            drop(svc);
            retire_local(engine, name, md, uuids, rep, &format!("another incarnation than [{}]: {why}", im.uuid))
        }
        Incarnation::Unknown(why) => {
            let path = svc.path().to_path_buf();
            drop(svc);
            if !holds_shard_data(&path) {
                tracing::info!(index = %name, "replacing a metadata-only local index of unknown identity: {why}");
                remove_index(engine, name, uuids, rep);
                return true;
            }
            quarantine_index(engine, name, &why, uuids, rep)
        }
    }
}

/// The cluster index uuids the local index `svc` is recorded under here:
/// the uuid in its own metadata file, the uuid map's entry for this exact
/// local index, its shard copy record. Empty: the index never took part in
/// a committed state on this node (a dry run, a rolled-back creation).
pub fn recorded_uuids(svc: &elasticsearch_engine::index::IndexService, uuids: &UuidMap) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |u: &str| if !out.iter().any(|o| o == u) { out.push(u.to_string()) };
    if let Some(c) = svc.cluster_uuid() {
        add(c);
    }
    if let Some((c, l)) = uuids.indices.get(&svc.name) {
        if l == &svc.uuid {
            add(c);
        }
    }
    if let Some(rec) = crate::shards::read_copy_record(svc.path()) {
        add(&rec.index_uuid);
    }
    out
}

/// What becomes of a local index the metadata does not hold (as this
/// incarnation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposal {
    /// Deleted: a tombstone names it, it holds no shard data, or it never
    /// was part of a committed state here.
    Delete(String),
    /// Kept, out of service, as a dangling index (ES `DanglingIndicesState`):
    /// listed by `GET _dangling`, importable or deletable through that API.
    Dangling(String),
}

/// ES: a local index the cluster metadata lacks is deleted only when the
/// index graveyard names it; otherwise it is dangling, never deleted.
pub fn disposal(svc: &elasticsearch_engine::index::IndexService, md: &Metadata, uuids: &UuidMap) -> Disposal {
    let recorded = recorded_uuids(svc, uuids);
    if let Some(u) = recorded.iter().chain(std::iter::once(&svc.uuid)).find(|u| elasticsearch_cluster::graveyard::contains(md, u)) {
        return Disposal::Delete(format!("a tombstone names index uuid [{u}]"));
    }
    if !holds_shard_data(svc.path()) {
        return Disposal::Delete("it holds no shard data".into());
    }
    // Only what an application of a committed state writes counts: the
    // uuid map also names indices a master's engine created for a
    // publication that may have failed (rolled back, never committed).
    let committed = svc.cluster_uuid().is_some() || crate::shards::read_copy_record(svc.path()).is_some();
    if !committed {
        return Disposal::Delete("it never was part of a committed cluster state on this node".into());
    }
    Disposal::Dangling(format!("no tombstone names index uuid {recorded:?}"))
}

/// Take the local index `name`, which `md` does not hold (as this
/// incarnation), out of service: deleted or kept dangling ([`disposal`]).
/// False when the name is still held (keeping it dangling failed).
fn retire_local(engine: &Engine, name: &str, md: &Metadata, uuids: &mut UuidMap, rep: &mut ApplyReport, context: &str) -> bool {
    let Some(svc) = engine.get_index(name) else { return true };
    let verdict = disposal(&svc, md, uuids);
    if matches!(verdict, Disposal::Dangling(_)) && svc.cluster_uuid().is_none() {
        // Listed (and importable) under its cluster uuid, not its local one.
        let from_record = crate::shards::read_copy_record(svc.path()).map(|r| r.index_uuid);
        if let Some(u) = from_record.or_else(|| recorded_uuids(&svc, uuids).into_iter().next()) {
            if let Err(e) = svc.set_cluster_uuid(&u) {
                tracing::warn!(index = %name, "cannot record the dangling copy's index uuid: {}", e.reason);
            }
        }
    }
    drop(svc);
    match verdict {
        Disposal::Delete(why) => {
            tracing::info!(index = %name, "deleting the local copy ({context}): {why}");
            remove_index(engine, name, uuids, rep);
            true
        }
        Disposal::Dangling(why) => match engine.quarantine_local_index_copy(name) {
            Ok(to) => {
                tracing::warn!(index = %name, kept = %to.display(), "local index not in the cluster metadata ({context}) kept as a dangling index: {why}");
                uuids.indices.remove(name);
                rep.dangling.push((name.to_string(), to));
                true
            }
            Err(e) => {
                rep.errors.push((name.to_string(), format!("cannot set aside dangling index ({context}, {why}): {}", e.reason)));
                false
            }
        },
    }
}

/// Delete set-aside copies (dangling or quarantined) whose index a
/// tombstone of `md` names (ES deletes a dangling index once tombstoned).
pub fn purge_tombstoned_set_aside(engine: &Engine, md: &Metadata) {
    if elasticsearch_cluster::graveyard::tombstones(md).is_empty() {
        return;
    }
    for (d, path) in crate::rest::dangling::quarantined_in(&engine.data) {
        if elasticsearch_cluster::graveyard::contains(md, d.uuid.as_str()) {
            match std::fs::remove_dir_all(&path) {
                Ok(()) => tracing::info!(index = %d.name, uuid = %d.uuid, "deleted a dangling copy its tombstone names"),
                Err(e) => tracing::warn!(index = %d.name, uuid = %d.uuid, "cannot delete a tombstoned dangling copy: {e}"),
            }
        }
    }
}

/// The set-aside copy (dangling or quarantined) of incarnation `uuid` of
/// `name` in `engine`, if any: re-registered as the engine's index (ES's
/// dangling import allocates the existing store).
fn adopt_set_aside(engine: &Engine, name: &str, uuid: &str) -> Option<std::sync::Arc<elasticsearch_engine::index::IndexService>> {
    let (_, path) = crate::rest::dangling::quarantined_in(&engine.data).into_iter().find(|(d, _)| d.uuid.as_str() == uuid && d.name.as_str() == name)?;
    match engine.adopt_set_aside_index_copy(name, &path) {
        Ok(svc) => {
            tracing::info!(index = %name, %uuid, from = %path.display(), "re-registered the dangling copy of the index");
            Some(svc)
        }
        Err(e) => {
            tracing::warn!(index = %name, %uuid, "cannot re-register the dangling copy: {}", e.reason);
            None
        }
    }
}

/// Move the local copy of `name` aside (see
/// `Engine::quarantine_local_index_copy`); it is listed by the dangling
/// indices API until an operator deletes it. False when it stays.
fn quarantine_index(engine: &Engine, name: &str, why: &str, uuids: &mut UuidMap, rep: &mut ApplyReport) -> bool {
    match engine.quarantine_local_index_copy(name) {
        Ok(to) => {
            tracing::error!(index = %name, kept = %to.display(), "local index copy of unknown identity quarantined (not deleted; listed by GET _dangling): {why}");
            uuids.indices.remove(name);
            rep.quarantined.push((name.to_string(), to));
            true
        }
        Err(e) => {
            rep.errors.push((name.to_string(), format!("cannot quarantine a local copy of unknown identity ({why}): {}", e.reason)));
            false
        }
    }
}

/// Build cluster metadata from the master's engine, carrying over the
/// cluster-owned parts of `prev` (uuids, primary terms, in-sync ids,
/// customs the engine does not know about) and bumping per-index versions
/// where something changed.
pub fn extract(engine: &Engine, prev: &Metadata, uuids: &mut UuidMap) -> Metadata {
    let mut md = prev.clone();
    md.indices.clear();
    for svc in engine.all_indices() {
        // An index the cluster does not know that took part in a committed
        // state (its copy records its cluster uuid or holds a shard copy
        // record) is dangling, not new: it is never published from here
        // (ES's master never adopts a local index; only an import does).
        if !prev.indices.contains_key(&svc.name) && (svc.cluster_uuid().is_some() || crate::shards::read_copy_record(svc.path()).is_some()) {
            tracing::debug!(index = %svc.name, "not publishing a dangling local index");
            continue;
        }
        let im = extract_index(engine, &svc, prev, uuids);
        md.indices.insert(svc.name.clone(), im);
    }
    uuids.indices.retain(|n, _| md.indices.contains_key(n));
    if let Err(e) = uuids.save() {
        // The pairs stay in memory and are saved again with the next
        // change; each copy also records its identity itself.
        tracing::error!("failed to persist the index uuid map: {e}");
    }
    extract_globals(engine, &mut md);
    crate::meta_extra::extract(engine, &mut md);
    if md != *prev {
        md.version = prev.version + 1;
    }
    md
}

/// One index of [`extract`]: its metadata as the engine holds it, with the
/// cluster-owned parts (uuid, primary terms, in-sync ids, creation date,
/// derived routing shards) and per-part versions carried over from `prev`.
/// Parts equal to `prev`'s are `prev`'s own shared values: no copy, and an
/// exact round trip (per-node settings keep `prev`'s when nothing else in
/// the settings changed). Records the uuid pair in `uuids` (not saved).
pub fn extract_index(engine: &Engine, svc: &elasticsearch_engine::index::IndexService, prev: &Metadata, uuids: &mut UuidMap) -> IndexMetadata {
    let meta = svc.meta();
    let settings = index_settings(svc);
    let mappings = if uuids.mapping_conflicts.get(&svc.name).is_some_and(|(path, _, attempts)| path == svc.path() && *attempts >= 3) {
        prev.indices.get(&svc.name).map(|im| (*im.mappings).clone()).unwrap_or_else(|| meta.mappings.to_json())
    } else { meta.mappings.to_json() };
    let state = if meta.open { IndexState::Open } else { IndexState::Close };
    // A map entry for another local index means the engine re-created the
    // index (a new incarnation) unless the copy itself records that it is
    // `prev`'s (a stale map: the identity is the copy's own).
    let old = prev.indices.get(&svc.name).filter(|p| match uuids.indices.get(&svc.name) {
        Some((_, l)) if l != &svc.uuid => incarnation(svc, &p.uuid, uuids) == Incarnation::Same,
        _ => true,
    });
    if let (None, Some(p)) = (old, prev.indices.get(&svc.name)) {
        if p.in_sync_allocation_ids.values().any(|ids| !ids.is_empty()) {
            tracing::warn!(index = %svc.name, old_uuid = %p.uuid, new_uuid = %svc.uuid, "index with in-sync copies was re-created under the same name; its previous data is dropped");
        }
    }
    // The first time the master sees an index, its local uuid becomes
    // the cluster uuid.
    let uuid = old.map(|o| o.uuid.clone()).unwrap_or_else(|| svc.uuid.clone());
    uuids.indices.insert(svc.name.clone(), (uuid.clone(), svc.uuid.clone()));
    // Each shard is its own engine index (see `shard_engines`); the
    // engine copy on the master carries the requested shard count.
    let number_of_shards = parse_u32(settings.get("index.number_of_shards"), 1).max(1);
    // 0 = derived by the master from `number_of_shards` (ES default).
    let routing_num_shards = parse_u32(settings.get("index.number_of_routing_shards"), 0);
    let number_of_replicas = parse_u32(settings.get("index.number_of_replicas"), 1);
    let mut im = IndexMetadata {
        name: svc.name.clone(),
        uuid,
        version: 1,
        mapping_version: 1,
        settings_version: 1,
        aliases_version: 1,
        state,
        settings: settings.into(),
        mappings: mappings.into(),
        aliases: meta.aliases.clone(),
        number_of_shards,
        number_of_replicas,
        routing_num_shards,
        primary_terms: vec![1; number_of_shards as usize],
        in_sync_allocation_ids: BTreeMap::new(),
        creation_date: svc.creation_date,
        lifecycle: elasticsearch_engine::ilm::index_lifecycle_state(engine, svc).map(elasticsearch_cluster::state::Shared::new),
    };
    if let Some(o) = old {
        im.primary_terms = o.primary_terms.clone();
        im.in_sync_allocation_ids = o.in_sync_allocation_ids.clone();
        im.creation_date = o.creation_date;
        let same_mappings = o.mappings == im.mappings;
        let same_settings = strip_per_node(&o.settings) == strip_per_node(&im.settings);
        let same_aliases = o.aliases == im.aliases;
        let same_lifecycle = o.lifecycle == im.lifecycle;
        im.mapping_version = o.mapping_version + u64::from(!same_mappings);
        im.settings_version = o.settings_version + u64::from(!same_settings);
        im.aliases_version = o.aliases_version + u64::from(!same_aliases);
        let changed = !same_mappings || !same_settings || !same_aliases || !same_lifecycle || o.state != im.state;
        im.version = o.version + u64::from(changed);
        if same_mappings {
            im.mappings = o.mappings.clone();
        }
        if same_settings {
            im.settings = o.settings.clone();
            // Derived by the master on allocation when not set explicitly.
            im.routing_num_shards = o.routing_num_shards;
        }
        if same_aliases {
            im.aliases = o.aliases.clone();
        }
        if same_lifecycle {
            im.lifecycle = o.lifecycle.clone();
        }
    }
    im
}

/// The non-index parts of [`extract`] that index-level metadata requests
/// change: templates, cluster settings, data streams.
pub fn extract_globals(engine: &Engine, md: &mut Metadata) {
    let t = engine.templates.read();
    md.index_templates = t.index.clone();
    md.component_templates = t.component.clone();
    md.legacy_templates = t.legacy.clone();
    drop(t);
    let (persistent, transient) = engine.cluster_settings.read().clone();
    md.persistent_settings = persistent;
    md.transient_settings = transient;
    let ds = serde_json::to_value(&*engine.data_streams.read()).unwrap_or(Value::Null);
    if ds.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        md.customs.remove(DATA_STREAMS_CUSTOM);
    } else {
        md.customs.insert(DATA_STREAMS_CUSTOM.into(), ds);
    }
}

/// Whether the parts [`extract_globals`] covers differ.
pub fn globals_differ(a: &Metadata, b: &Metadata) -> bool {
    a.index_templates != b.index_templates || a.component_templates != b.component_templates || a.legacy_templates != b.legacy_templates
        || a.persistent_settings != b.persistent_settings || a.transient_settings != b.transient_settings
        || a.customs.get(DATA_STREAMS_CUSTOM) != b.customs.get(DATA_STREAMS_CUSTOM)
}

/// What a batch of engine metadata changes did to the metadata: the
/// entries that differ between the state the engine was extracted against
/// (`base`) and the extraction (`ours`). Applying it to a newer state keeps
/// every entry the batch did not touch as that state has it.
#[derive(Clone, Debug, Default)]
pub struct MetadataDelta {
    /// What every touched entry looked like in the metadata the delta was
    /// computed against: the published task applies the delta only if the
    /// current state still matches (compare-and-set).
    pub expect: Expectations,
    pub indices: BTreeMap<String, Option<IndexMetadata>>,
    pub index_templates: BTreeMap<String, Option<Value>>,
    pub component_templates: BTreeMap<String, Option<Value>>,
    pub legacy_templates: BTreeMap<String, Option<Value>>,
    pub persistent_settings: Option<Map<String, Value>>,
    pub transient_settings: Option<Map<String, Value>>,
    pub customs: BTreeMap<String, Option<Value>>,
}

/// The engine-relevant identity of an index's metadata: what a change
/// computed against it assumed. Routing-owned parts (in-sync ids, primary
/// terms) are not part of it: they are re-read from the live state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexStamp {
    pub uuid: String,
    pub state: IndexState,
    pub mapping_version: u64,
    pub settings_version: u64,
    pub aliases_version: u64,
    pub lifecycle: u64,
}

impl IndexStamp {
    pub fn of(im: &IndexMetadata) -> IndexStamp {
        IndexStamp {
            uuid: im.uuid.clone(),
            state: im.state,
            mapping_version: im.mapping_version,
            settings_version: im.settings_version,
            aliases_version: im.aliases_version,
            lifecycle: im.lifecycle.as_deref().map(value_hash).unwrap_or(0),
        }
    }
}

fn value_hash<T: serde::Serialize + ?Sized>(v: &T) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(v).unwrap_or_default().hash(&mut h);
    h.finish()
}

/// The base a [`MetadataDelta`] was computed against, for every entry it
/// touches (`None`: absent from the base).
#[derive(Clone, Debug, Default)]
pub struct Expectations {
    pub indices: BTreeMap<String, Option<IndexStamp>>,
    /// `<kind>/<key>` of templates, customs and the two settings maps →
    /// hash of the base value (`None`: absent).
    pub globals: BTreeMap<String, Option<u64>>,
}

fn global_entries(md: &Metadata, delta: &MetadataDelta) -> BTreeMap<String, Option<u64>> {
    let mut out = BTreeMap::new();
    for (kind, d, m) in [("index_template", &delta.index_templates, &md.index_templates), ("component_template", &delta.component_templates, &md.component_templates), ("legacy_template", &delta.legacy_templates, &md.legacy_templates), ("custom", &delta.customs, &md.customs)] {
        for k in d.keys() {
            out.insert(format!("{kind}/{k}"), m.get(k).map(value_hash));
        }
    }
    if delta.persistent_settings.is_some() {
        out.insert("persistent_settings".into(), Some(value_hash(&md.persistent_settings)));
    }
    if delta.transient_settings.is_some() {
        out.insert("transient_settings".into(), Some(value_hash(&md.transient_settings)));
    }
    out
}

impl Expectations {
    /// Whether `md` still is the base (for the entries the delta touches);
    /// the first mismatch otherwise.
    pub fn check(&self, md: &Metadata) -> Result<(), String> {
        for (name, want) in &self.indices {
            let have = md.indices.get(name).map(IndexStamp::of);
            if &have != want {
                return Err(format!("index [{name}] changed concurrently (expected {want:?}, found {have:?})"));
            }
        }
        for (key, want) in &self.globals {
            let have = global_value_hash(md, key);
            if &have != want {
                return Err(format!("[{key}] changed concurrently"));
            }
        }
        Ok(())
    }
}

fn global_value_hash(md: &Metadata, key: &str) -> Option<u64> {
    match key {
        "persistent_settings" => Some(value_hash(&md.persistent_settings)),
        "transient_settings" => Some(value_hash(&md.transient_settings)),
        _ => {
            let (kind, k) = key.split_once('/')?;
            let m = match kind {
                "index_template" => &md.index_templates,
                "component_template" => &md.component_templates,
                "legacy_template" => &md.legacy_templates,
                _ => &md.customs,
            };
            m.get(k).map(value_hash)
        }
    }
}

fn map_delta<V: Clone + PartialEq>(base: &BTreeMap<String, V>, ours: &BTreeMap<String, V>) -> BTreeMap<String, Option<V>> {
    let mut d = BTreeMap::new();
    for (k, v) in ours {
        if base.get(k) != Some(v) {
            d.insert(k.clone(), Some(v.clone()));
        }
    }
    for k in base.keys() {
        if !ours.contains_key(k) {
            d.insert(k.clone(), None);
        }
    }
    d
}

fn apply_map<V: Clone>(d: &BTreeMap<String, Option<V>>, m: &mut BTreeMap<String, V>) {
    for (k, v) in d {
        match v {
            Some(v) => {
                m.insert(k.clone(), v.clone());
            }
            None => {
                m.remove(k);
            }
        }
    }
}

impl MetadataDelta {
    pub fn between(base: &Metadata, ours: &Metadata) -> MetadataDelta {
        MetadataDelta {
            expect: Expectations::default(),
            indices: map_delta(&base.indices, &ours.indices),
            index_templates: map_delta(&base.index_templates, &ours.index_templates),
            component_templates: map_delta(&base.component_templates, &ours.component_templates),
            legacy_templates: map_delta(&base.legacy_templates, &ours.legacy_templates),
            persistent_settings: (base.persistent_settings != ours.persistent_settings).then(|| ours.persistent_settings.clone()),
            transient_settings: (base.transient_settings != ours.transient_settings).then(|| ours.transient_settings.clone()),
            customs: map_delta(&base.customs, &ours.customs),
        }
    }

    /// Record `base` as what every entry this delta touches must still be
    /// when it is applied (see [`Expectations`]).
    pub fn expect_base(&mut self, base: &Metadata) {
        self.expect.indices = self.indices.keys().map(|k| (k.clone(), base.indices.get(k).map(IndexStamp::of))).collect();
        self.expect.globals = global_entries(base, self);
    }

    /// The globals' expectations only, against `base`.
    pub fn expect_globals_base(&mut self, base: &Metadata) {
        self.expect.globals = global_entries(base, self);
    }

    /// The templates, cluster settings and data streams that differ
    /// between `base` and `after` (see [`extract_globals`]).
    pub fn set_globals(&mut self, base: &Metadata, after: &Metadata) {
        self.index_templates = map_delta(&base.index_templates, &after.index_templates);
        self.component_templates = map_delta(&base.component_templates, &after.component_templates);
        self.legacy_templates = map_delta(&base.legacy_templates, &after.legacy_templates);
        self.persistent_settings = (base.persistent_settings != after.persistent_settings).then(|| after.persistent_settings.clone());
        self.transient_settings = (base.transient_settings != after.transient_settings).then(|| after.transient_settings.clone());
        let (b, a) = (base.customs.get(DATA_STREAMS_CUSTOM), after.customs.get(DATA_STREAMS_CUSTOM));
        if b != a {
            self.customs.insert(DATA_STREAMS_CUSTOM.to_string(), a.cloned());
        }
    }

    /// Whether anything besides indices changes.
    pub fn has_globals(&self) -> bool {
        !self.index_templates.is_empty() || !self.component_templates.is_empty() || !self.legacy_templates.is_empty()
            || self.persistent_settings.is_some() || self.transient_settings.is_some() || !self.customs.is_empty()
    }

    /// Keep only what index-level metadata requests change: indices,
    /// templates, cluster settings and data streams (a scratch copy's other
    /// customs are validation inputs, not changes).
    pub fn retain_index_level(&mut self) {
        self.customs.retain(|k, _| k == DATA_STREAMS_CUSTOM);
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.index_templates.is_empty() && self.component_templates.is_empty() && self.legacy_templates.is_empty()
            && self.persistent_settings.is_none() && self.transient_settings.is_none() && self.customs.is_empty()
    }

    pub fn apply_to(&self, md: &mut Metadata) {
        apply_map(&self.indices, &mut md.indices);
        self.apply_globals_to(md);
    }

    /// [`apply_to`](Self::apply_to) without the indices.
    pub fn apply_globals_to(&self, md: &mut Metadata) {
        apply_map(&self.index_templates, &mut md.index_templates);
        apply_map(&self.component_templates, &mut md.component_templates);
        apply_map(&self.legacy_templates, &mut md.legacy_templates);
        if let Some(p) = &self.persistent_settings {
            md.persistent_settings = p.clone();
        }
        if let Some(t) = &self.transient_settings {
            md.transient_settings = t.clone();
        }
        apply_map(&self.customs, &mut md.customs);
    }
}

fn strip_per_node(s: &Map<String, Value>) -> Map<String, Value> {
    s.iter().filter(|(k, _)| !PER_NODE_SETTINGS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// The engine create-index body that reproduces `im` on another node. The
/// creation version travels with it: version-keyed behaviour (ES 9's
/// modulo shard routing, mapping defaults) must match on every copy.
pub fn create_body(im: &IndexMetadata) -> Value {
    let mut settings = strip_per_node(&im.settings);
    // These are cluster metadata in ES: every node reports the index's own
    // creation date, name and version (not the time this copy was made).
    for key in ["index.version.created", "index.creation_date", "index.provided_name"] {
        if let Some(v) = im.settings.get(key) {
            settings.insert(key.into(), v.clone());
        }
    }
    json!({
        "settings": Value::Object(settings),
        "mappings": im.mappings,
        "aliases": im.aliases,
    })
}

/// Create `im`'s local mirror as a *sparse* engine index: metadata only,
/// no shard copy open. The shard layer opens exactly the copies the
/// routing table assigns to this node (`node::ensure_copy_open`), so an
/// engine never holds empty copies of shards that live elsewhere.
///
/// The local uuid is the cluster uuid when the engine accepts it (it must
/// be a valid, unused engine uuid); otherwise a fresh one. Either way the
/// caller records the pair in the [`UuidMap`].
pub fn create_sparse(engine: &Engine, name: &str, im: &IndexMetadata) -> Result<std::sync::Arc<elasticsearch_engine::index::IndexService>, elasticsearch_engine::error::EsError> {
    let mut body = create_body(im);
    // Creation uses the public validator. Install inert archived entries only
    // through the committed-metadata path after the sparse shell exists.
    let archived: FlatSettings = im.settings.iter().filter(|(key, _)| key.starts_with("archived."))
        .map(|(key, value)| (key.clone(), value.clone())).collect();
    body["settings"].as_object_mut().unwrap().retain(|key, _| !key.starts_with("archived."));
    let service = match engine.create_index_sparse(name, &im.uuid, &body) {
        Ok(s) => Ok(s),
        Err(e) if e.kind == "illegal_argument_exception" && e.reason.contains("uuid") => engine.create_index_sparse(name, &elasticsearch_engine::index::new_uuid(), &body),
        Err(e) => Err(e),
    }?;
    if !archived.is_empty() { service.apply_settings_delta(archived)?; }
    Ok(service)
}

/// What `apply` changed (for logging and for the shard layer, which must
/// drop local shard state for deleted/recreated indices).
#[derive(Default, Debug)]
pub struct ApplyReport {
    pub created: Vec<String>,
    pub deleted: Vec<String>,
    pub updated: Vec<String>,
    pub errors: Vec<(String, String)>,
    pub failed_mappings: Vec<(String, String)>,
    /// Indices of the state left absent from the engine because an engine
    /// API in flight deleted them there (see [`HeldDeletes`]): their delete
    /// is published by the API's kept job, not undone by this application.
    pub withheld: Vec<String>,
    /// Local indices whose identity was missing from the uuid map and was
    /// reconstructed from the copy itself.
    pub reconstructed: Vec<String>,
    /// Local copies of unknown identity moved aside (name, new location).
    pub quarantined: Vec<(String, PathBuf)>,
    /// Local indices the metadata does not hold, kept as dangling indices
    /// (name, new location).
    pub dangling: Vec<(String, PathBuf)>,
}

/// Index incarnations (name, cluster uuid) that an engine API in flight on
/// the master's engine 0 held when it took its `before` snapshot. While it
/// is in flight, an application never re-creates one of them that engine 0
/// no longer holds: that absence is the API's delete, which its kept job
/// publishes (diffed against this state); re-creating the shell would turn
/// the delete into "no change" and leave a stale shell under a held primary.
pub type HeldDeletes = std::collections::BTreeSet<(String, String)>;

/// Make the local engine's metadata match `md`.
pub fn apply(engine: &Engine, md: &Metadata, uuids: &mut UuidMap) -> ApplyReport {
    apply_scoped(engine, md, uuids, true)
}

/// [`apply`]; without `extras` only indices, templates, cluster settings
/// and data streams (what the index-level metadata APIs change: the master
/// service undoes a dry run this way), not the resources `meta_extra` keeps.
pub fn apply_scoped(engine: &Engine, md: &Metadata, uuids: &mut UuidMap, extras: bool) -> ApplyReport {
    let mut rep = ApplyReport::default();
    // ILM policies, pipelines, synonym sets (before indices that use them).
    if extras {
        rep.errors.extend(crate::meta_extra::apply(engine, md));
    }
    // Deleted indices, and indices recreated under the same name. A local
    // index the map does not (correctly) name is identified from the copy
    // itself, never deleted for want of a map entry.
    let mut blocked = std::collections::BTreeSet::new();
    for svc in engine.all_indices() {
        let name = svc.name.clone();
        drop(svc);
        match md.indices.get(&name) {
            Some(im) => {
                if !reconcile_incarnation(engine, &name, im, md, uuids, &mut rep) {
                    blocked.insert(name);
                }
            }
            None => {
                retire_local(engine, &name, md, uuids, &mut rep, "absent from the cluster metadata");
            }
        }
    }
    purge_tombstoned_set_aside(engine, md);
    for (name, im) in &md.indices {
        if !blocked.contains(name) {
            apply_index(engine, name, im, uuids, &mut rep);
        }
    }
    apply_globals(engine, md);
    if let Err(e) = engine.persist_state() {
        rep.errors.push(("_state".into(), e.reason.clone()));
    }
    if let Err(e) = uuids.save() {
        rep.errors.push((UUIDS_FAILURE.into(), format!("failed to persist the index uuid map: {e}")));
    }
    rep
}

/// The [`ApplyReport::errors`] key of a uuid map that could not be saved
/// (not an index: the next application retries it with the globals).
pub const UUIDS_FAILURE: &str = "_uuids";

/// [`apply_scoped`] for a state whose predecessor `prev` the engine already
/// matches: only indices whose metadata changed (or that the engine lacks)
/// are touched, the globals only if they changed. Unchanged index metadata
/// is shared between states, so this is O(changed) work per application.
pub fn apply_changes(engine: &Engine, prev: &Metadata, md: &Metadata, uuids: &mut UuidMap, extras: bool, retry: &std::collections::BTreeSet<String>) -> ApplyReport {
    apply_changes_holding(engine, prev, md, uuids, extras, retry, &HeldDeletes::new())
}

/// [`apply_changes`] that leaves absent the incarnations in `held` an engine
/// API in flight has deleted (reported in [`ApplyReport::withheld`]).
pub fn apply_changes_holding(engine: &Engine, prev: &Metadata, md: &Metadata, uuids: &mut UuidMap, extras: bool, retry: &std::collections::BTreeSet<String>, held: &HeldDeletes) -> ApplyReport {
    let mut rep = ApplyReport::default();
    // A failure that named no index (scripts, pipelines, ...) is retried by
    // applying the globals and customs again.
    let retry_globals = retry.iter().any(|n| !md.indices.contains_key(n) && !prev.indices.contains_key(n));
    if extras && retry_globals {
        rep.errors.extend(crate::meta_extra::apply(engine, md));
    } else if extras && prev.customs != md.customs {
        // Entry by entry: an engine API's unpublished change survives.
        rep.errors.extend(crate::meta_extra::apply_changed(engine, prev, md));
    }
    for name in prev.indices.keys() {
        // Deleted from the cluster (a name replaced by another incarnation
        // is reconciled below, from what the local copy is): deleted when
        // tombstoned, else kept dangling.
        if !md.indices.contains_key(name) && engine.get_index(name).is_some() {
            retire_local(engine, name, md, uuids, &mut rep, "deleted from the cluster metadata");
        }
    }
    if prev.customs.get(elasticsearch_cluster::graveyard::CUSTOM) != md.customs.get(elasticsearch_cluster::graveyard::CUSTOM) {
        purge_tombstoned_set_aside(engine, md);
    }
    for (name, im) in &md.indices {
        // Mapped: the map names this incarnation for the local index there
        // is (a stale entry, for another local index, is not).
        let local = engine.get_index(name);
        let mapped = match (&local, uuids.indices.get(name)) {
            (Some(svc), Some((cluster, l))) => cluster == &im.uuid && l == &svc.uuid,
            (None, Some((cluster, _))) => cluster == &im.uuid,
            (_, None) => false,
        };
        let present = local.is_some();
        drop(local);
        let same = prev.indices.get(name).is_some_and(|p| same_engine_view(p, im));
        // `retry`: indices whose last application failed are applied again.
        if !same || !mapped || retry.contains(name) || !present {
            // Another incarnation under this name is removed; one of
            // unknown identity is quarantined (and blocks the name while
            // it cannot be).
            if !reconcile_incarnation(engine, name, im, md, uuids, &mut rep) {
                continue;
            }
            // The incarnation an engine API in flight saw: never created
            // here (also not if the API deletes it while this runs).
            let create = held.is_empty() || !held.contains(&(name.clone(), im.uuid.clone()));
            apply_index_inner(engine, name, im, uuids, &mut rep, create);
        }
    }
    let globals = globals_differ(prev, md) || retry_globals;
    if globals {
        apply_globals(engine, md);
    }
    if globals || !rep.created.is_empty() || !rep.deleted.is_empty() || !rep.updated.is_empty() || !rep.reconstructed.is_empty() || !rep.quarantined.is_empty() || !rep.dangling.is_empty() {
        if let Err(e) = engine.persist_state() {
            rep.errors.push(("_state".into(), e.reason.clone()));
        }
        if let Err(e) = uuids.save() {
            rep.errors.push((UUIDS_FAILURE.into(), format!("failed to persist the index uuid map: {e}")));
        }
    }
    rep
}

/// Whether two metadata of an index agree on everything an engine copy
/// holds (routing-owned parts such as in-sync ids do not count).
fn same_engine_view(a: &IndexMetadata, b: &IndexMetadata) -> bool {
    a.uuid == b.uuid && a.state == b.state && a.settings == b.settings && a.mappings == b.mappings && a.aliases == b.aliases && a.lifecycle == b.lifecycle
}

fn remove_index(engine: &Engine, name: &str, uuids: &mut UuidMap, rep: &mut ApplyReport) {
    match crate::meta_extra::delete_local_copy(engine, name) {
        Ok(()) => rep.deleted.push(name.to_string()),
        Err(e) => rep.errors.push((name.to_string(), e)),
    }
    uuids.indices.remove(name);
}

/// Make one index of the local engine match `im` (create it as a sparse
/// shell if missing; settings, mappings, aliases, open state otherwise).
/// The caller removed a copy of another incarnation first.
pub fn apply_index(engine: &Engine, name: &str, im: &IndexMetadata, uuids: &mut UuidMap, rep: &mut ApplyReport) {
    apply_index_inner(engine, name, im, uuids, rep, true)
}

/// [`apply_index`]; without `create`, an index the engine lacks is left
/// absent and reported in [`ApplyReport::withheld`].
fn apply_index_inner(engine: &Engine, name: &str, im: &IndexMetadata, uuids: &mut UuidMap, rep: &mut ApplyReport, create: bool) {
    let name = &name.to_string();
    let svc = match engine.get_index(name) {
        Some(s) => s,
        None if !create => {
            rep.withheld.push(name.clone());
            return;
        }
        None => match adopt_set_aside(engine, name, &im.uuid).map(Ok).unwrap_or_else(|| create_sparse(engine, name, im)) {
            Ok(s) => {
                uuids.indices.insert(name.clone(), (im.uuid.clone(), s.uuid.clone()));
                rep.created.push(name.clone());
                s
            }
            Err(e) => {
                rep.errors.push((name.clone(), e.reason.clone()));
                return;
            }
        },
    };
    if uuids.mapping_conflicts.get(name).is_some_and(|(path, _, attempts)| path == svc.path() && *attempts >= 3) {
        // Recovery replaces the quarantined incarnation; do not retry an
        // immutable mapper conflict on the same service forever.
        return;
    }
    // The copy's incarnation, readable without the uuid maps (and durable
    // in its metadata file: the identity survives a lost map).
    if uuids.indices.get(name).is_some_and(|(cluster, local)| cluster == &im.uuid && local == &svc.uuid) {
        if let Err(e) = svc.set_cluster_uuid(&im.uuid) {
            rep.errors.push((name.clone(), format!("failed to persist the copy's index uuid: {}", e.reason)));
        }
    }
    let mut changed = false;
    let want_open = im.state == IndexState::Open;
    if !want_open && svc.is_open() {
        changed |= svc.set_open(false).is_ok();
        // Pinned query_then_fetch views must not outlive the open index.
        engine.drop_fetch_contexts(name);
    }
    let local = svc.meta();
    // Settings: send only differing keys; removed keys as null.
    let have = strip_per_node(&index_settings(&svc));
    let want = strip_per_node(&im.settings);
    let mut delta: FlatSettings = FlatSettings::new();
    for (k, v) in &want {
        if have.get(k) != Some(v) {
            delta.insert(k.clone(), v.clone());
        }
    }
    for k in have.keys() {
        if !want.contains_key(k) {
            delta.insert(k.clone(), Value::Null);
        }
    }
    if !delta.is_empty() {
        // Non-dynamic keys can only change while closed; the master
        // already validated the change, so briefly close if needed.
        let needs_close = svc.is_open() && delta.keys().any(|k| !es_settings::is_dynamic(k) && !es_settings::is_private_dynamic(k));
        if needs_close {
            let _ = svc.set_open(false);
            engine.drop_fetch_contexts(name);
        }
        match svc.apply_settings_delta(delta) {
            Ok(()) => changed = true,
            Err(e) => rep.errors.push((name.clone(), e.reason.clone())),
        }
        if needs_close {
            let _ = svc.set_open(true);
        }
    }
    // A primary's replication request may already have installed a newer
    // committed mapping on this copy: an older state never reverts it.
    let newer_here = svc.mapping_version() > im.mapping_version;
    if newer_here {
        tracing::debug!(index = %name, "the copy's mapping is newer than this state's; kept");
    } else if local.mappings.to_json() != *im.mappings {
        match svc.put_mapping(&im.mappings) {
            Ok(()) => {
                uuids.mapping_conflicts.remove(name);
                changed = true;
                svc.advance_mapping_version(im.mapping_version);
            }
            Err(e) => {
                if matches!(e.kind.as_str(), "illegal_argument_exception" | "mapper_parsing_exception") {
                    let failure = uuids.mapping_conflicts.entry(name.clone()).or_insert((svc.path().to_path_buf(), im.mapping_version, 0));
                    if failure.0 != svc.path() { *failure = (svc.path().to_path_buf(), im.mapping_version, 0); }
                    failure.1 = im.mapping_version;
                    failure.2 += 1;
                    if failure.2 >= 3 {
                        svc.quarantine_mapping();
                        rep.failed_mappings.push((name.clone(), e.reason.clone()));
                        return;
                    }
                }
                rep.errors.push((name.clone(), e.reason.clone()));
            }
        }
    } else {
        uuids.mapping_conflicts.remove(name);
        svc.advance_mapping_version(im.mapping_version);
    }
    if local.aliases != im.aliases {
        match svc.set_aliases(im.aliases.clone()) {
            Ok(()) => changed = true,
            Err(e) => rep.errors.push((name.clone(), e.reason.clone())),
        }
    }
    if want_open && !svc.is_open() {
        changed |= svc.set_open(true).is_ok();
    }
    // Lifecycle execution state (rewritten only when it differs). An absent
    // state does not clear one this copy holds while the index is still
    // lifecycle-managed, nor its rollover info (never removed in ES): a
    // master from before the state was published (a rolling upgrade)
    // omits it, and must not wipe it.
    let local = if im.lifecycle.is_none() { elasticsearch_engine::ilm::index_lifecycle_state(engine, &svc) } else { None };
    let managed = svc.setting("index.lifecycle.name").is_some_and(|p| !p.is_empty());
    let want: Option<Value> = match (&im.lifecycle, local) {
        (Some(v), _) => Some((**v).clone()),
        // Still managed: the execution state stays.
        (None, Some(local)) if managed => Some(local),
        // Not managed: only the rollover info stays.
        (None, Some(local)) if local["rollover_info"].as_object().is_some_and(|o| !o.is_empty()) => Some(json!({"rollover_info": local["rollover_info"]})),
        (None, _) => None,
    };
    if let Err(e) = elasticsearch_engine::ilm::restore_index_lifecycle_state(engine, &svc, want.as_ref()) {
        rep.errors.push((name.clone(), e.reason.clone()));
    }
    if changed && !rep.created.contains(name) {
        rep.updated.push(name.clone());
    }
}

/// Templates, cluster settings and data streams from `md`.
pub fn apply_globals(engine: &Engine, md: &Metadata) {
    {
        let mut t = engine.templates.write();
        t.index = md.index_templates.clone();
        t.component = md.component_templates.clone();
        t.legacy = md.legacy_templates.clone();
    }
    *engine.cluster_settings.write() = (md.persistent_settings.clone(), md.transient_settings.clone());
    elasticsearch_engine::memory::configure(engine);
    if let Err(error) = elasticsearch_engine::script::rate_limit::configure(engine) {
        tracing::warn!(%error, "invalid script compilation rate in applied metadata");
    }
    if let Some(ds) = md.customs.get(DATA_STREAMS_CUSTOM) {
        if let Ok(ds) = serde_json::from_value(ds.clone()) {
            *engine.data_streams.write() = ds;
        }
    } else {
        engine.data_streams.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_mapping_conflict_is_quarantined_after_three_attempts() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let svc = engine.create_index("bad", &json!({"mappings":{"properties":{"c":{"type":"constant_keyword"}}}})).unwrap();
        let mut uuids = UuidMap::default();
        let committed = extract(&engine, &Metadata::default(), &mut uuids);
        svc.put_mapping(&json!({"properties":{"c":{"type":"constant_keyword","value":"uncommitted"}}})).unwrap();
        for attempt in 1..=5 {
            let mut report = ApplyReport::default();
            apply_index(&engine, "bad", &committed.indices["bad"], &mut uuids, &mut report);
            assert_eq!(report.errors.len(), usize::from(attempt < 3));
            assert_eq!(report.failed_mappings.len(), usize::from(attempt == 3));
        }
        assert!(svc.is_detached());
        let extracted = extract(&engine, &committed, &mut uuids);
        assert_eq!(extracted.indices["bad"].mappings, committed.indices["bad"].mappings, "a quarantined local mapping must not be republished");
    }

    /// An application while an engine API that deleted an index is in
    /// flight: without the hold it re-creates the shell (and the API's
    /// before/after diff then sees no delete); with it the index stays
    /// absent and the diff is the delete.
    #[test]
    fn held_delete_is_not_recreated_by_an_application() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        engine.create_index("s", &json!({})).unwrap();
        let mut uuids = UuidMap::default();
        let md = extract(&engine, &Metadata::default(), &mut uuids);
        let cluster_uuid = md.indices["s"].uuid.clone();
        assert_eq!(uuids.cluster_uuid("s"), Some(cluster_uuid.as_str()));
        let before = crate::scratch::snapshot_of(&engine, false);
        let none = std::collections::BTreeSet::new();

        engine.delete_index("s").unwrap();
        let rep = apply_changes(&engine, &md, &md, &mut uuids, false, &none);
        assert_eq!(rep.created, vec!["s".to_string()], "the unheld application re-creates the deleted shell");
        let after = crate::scratch::snapshot_of(&engine, false);
        assert_ne!(crate::scratch::changes_between(&engine, &mut uuids, &[&before], &after, &md).get("s"), Some(&None), "the delete is lost");

        engine.delete_index("s").unwrap();
        let held: HeldDeletes = [("s".to_string(), cluster_uuid.clone())].into_iter().collect();
        let rep = apply_changes_holding(&engine, &md, &md, &mut uuids, false, &none, &held);
        assert!(rep.created.is_empty() && rep.errors.is_empty());
        assert_eq!(rep.withheld, vec!["s".to_string()]);
        assert!(engine.get_index("s").is_none());
        let after = crate::scratch::snapshot_of(&engine, false);
        let changes = crate::scratch::changes_between(&engine, &mut uuids, &[&before], &after, &md);
        assert_eq!(changes.get("s"), Some(&None), "the API's delete is what it publishes");

        // Another incarnation under the name (deleted and re-created by the
        // state meanwhile) is not held.
        let mut recreated = md.clone();
        recreated.indices.get_mut("s").unwrap().uuid = "another-incarnation".into();
        let rep = apply_changes_holding(&engine, &md, &recreated, &mut uuids, false, &none, &held);
        assert!(rep.withheld.is_empty());
        assert!(engine.get_index("s").is_some());
    }

    #[test]
    fn delta_carries_only_what_the_batch_changed() {
        let mut base = Metadata::default();
        base.customs.insert("kept".into(), json!({"a": 1}));
        base.customs.insert("gone".into(), json!(1));
        base.index_templates.insert("t".into(), json!({"x": 1}));
        let mut ours = base.clone();
        ours.customs.remove("gone");
        ours.customs.insert("new".into(), json!(2));
        ours.persistent_settings.insert("cluster.x".into(), json!("1"));
        let d = MetadataDelta::between(&base, &ours);
        assert!(!d.is_empty());
        assert!(d.index_templates.is_empty() && d.transient_settings.is_none());
        // A newer state: another task changed `kept` and added a template.
        let mut live = base.clone();
        live.customs.insert("kept".into(), json!({"a": 2}));
        live.index_templates.insert("t2".into(), json!({}));
        d.apply_to(&mut live);
        assert_eq!(live.customs["kept"], json!({"a": 2}), "untouched entries keep the newer value");
        assert!(!live.customs.contains_key("gone"));
        assert_eq!(live.customs["new"], json!(2));
        assert!(live.index_templates.contains_key("t2"));
        assert_eq!(live.persistent_settings["cluster.x"], "1");
        assert!(MetadataDelta::between(&base, &base).is_empty());
    }

    #[test]
    fn copies_on_other_nodes_keep_the_cluster_level_index_settings() {
        let mut settings = Map::new();
        for (k, v) in [("index.uuid", "u1"), ("index.creation_date", "1792007643003"), ("index.provided_name", "cdx"), ("index.version.created", "9111000"), ("index.number_of_shards", "1")] {
            settings.insert(k.into(), Value::String(v.into()));
        }
        let im = IndexMetadata { name: "cdx".into(), uuid: "u1".into(), version: 1, mapping_version: 1, settings_version: 1, aliases_version: 1,
            state: IndexState::Open, settings: settings.into(), mappings: serde_json::json!({}).into(), aliases: BTreeMap::new(), number_of_shards: 1, number_of_replicas: 1,
            routing_num_shards: 1, primary_terms: vec![1], in_sync_allocation_ids: BTreeMap::new(), creation_date: 1792007643003, lifecycle: None };
        let body = create_body(&im);
        let s = &body["settings"];
        assert_eq!(s["index.creation_date"], "1792007643003");
        assert_eq!(s["index.provided_name"], "cdx");
        assert_eq!(s["index.version.created"], "9111000");
        // The engine uuid is chosen when the copy is created, not copied.
        assert!(s.get("index.uuid").is_none());
    }

    #[test]
    fn archived_metadata_applies_to_existing_and_new_sparse_mirrors() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        engine.create_index("legacy", &json!({})).unwrap();
        let mut uuids = UuidMap::default();
        let mut md = extract(&engine, &Metadata::default(), &mut uuids);
        md.indices.get_mut("legacy").unwrap().settings.insert("archived.index.max_regex_length".into(), json!("broken"));
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        assert_eq!(engine.get_index("legacy").unwrap().meta().settings["archived.index.max_regex_length"], "broken");
        let fresh = tempfile::tempdir().unwrap();
        let peer = Engine::open(fresh.path()).unwrap();
        let rep = apply(&peer, &md, &mut UuidMap::default());
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        assert_eq!(peer.get_index("legacy").unwrap().meta().settings["archived.index.max_regex_length"], "broken");
    }

    /// Closing an index through the metadata apply drops its
    /// query_then_fetch reader contexts.
    #[test]
    fn closing_through_the_apply_drops_fetch_contexts() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let svc = engine.create_index("cdx", &json!({"settings": {"number_of_shards": 1}})).unwrap();
        let mut uuids = UuidMap::default();
        let mut md = extract(&engine, &Metadata::default(), &mut uuids);
        let owner = elasticsearch_engine::node::FetchOwner { principal: None, index: None, shard: None };
        let pinned = elasticsearch_engine::search::pin_views(&[elasticsearch_engine::search::Target::new(svc.clone())]).unwrap();
        assert!(engine.register_fetch_context("q".into(), pinned, owner));
        md.indices.get_mut("cdx").unwrap().state = IndexState::Close;
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        assert!(!svc.is_open());
        assert!(engine.fetch_contexts.lock().is_empty());
    }

    // ---- Index identity across restarts with a damaged uuid map ----

    use elasticsearch_engine::index::{RecoverySource as EngineRecovery, ShardOpenSpec, WriteReq};
    use elasticsearch_engine::shard::WriteCond;

    /// Cluster metadata of a one-shard index `name` with cluster uuid
    /// `uuid` (one the engine refuses as its own when it holds a `.`, so the
    /// local uuid differs and the map matters).
    fn cluster_md(name: &str, uuid: &str) -> Metadata {
        let dir = tempfile::tempdir().unwrap();
        let master = Engine::open(dir.path()).unwrap();
        master.create_index(name, &json!({"settings": {"number_of_shards": 1}})).unwrap();
        let mut md = extract(&master, &Metadata::default(), &mut UuidMap::default());
        md.indices.get_mut(name).unwrap().uuid = uuid.into();
        md
    }

    /// A node's engine + map as at its start.
    fn start(dir: &Path) -> (std::sync::Arc<Engine>, UuidMap) {
        let engine = Engine::open(&dir.join("engine")).unwrap();
        let uuids = UuidMap::load_at_start(&dir.join("uuids.json")).unwrap();
        (engine, uuids)
    }

    /// Mirror `md` on a fresh node at `dir`, open its shard 0 copy, write
    /// `docs` documents and flush; with `record`, the copy record a started
    /// copy has. Returns the local uuid.
    fn node_with_data(dir: &Path, md: &Metadata, name: &str, docs: usize, record: bool) -> String {
        let (engine, mut uuids) = start(dir);
        let rep = apply(&engine, md, &mut uuids);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        let svc = engine.get_index(name).unwrap();
        let shard = svc.open_shard(0, ShardOpenSpec { allocation_id: "aid".into(), primary: true, primary_term: 1, recovery: EngineRecovery::Empty }).unwrap();
        for i in 0..docs {
            let r = svc.write(vec![WriteReq::Index { id: Some(format!("d{i}")), source: format!(r#"{{"n":{i}}}"#), routing: None, cond: WriteCond::default() }], false);
            assert!(r[0].is_ok(), "{:?}", r[0]);
        }
        shard.shard.flush().unwrap();
        if record {
            crate::shards::write_copy_record(&svc, &crate::shards::CopyRecord { index: name.into(), index_uuid: md.indices[name].uuid.clone(), shard: 0,
                allocation_id: "aid".into(), primary_term: 1, primary: true, history_uuid: Some("h".into()), global_checkpoint: Some(-1) }).unwrap();
        }
        svc.uuid.clone()
    }

    /// After a restart: the copy `local` is still registered under `name`
    /// and opens with every document it had.
    fn assert_intact(engine: &Engine, name: &str, local: &str, docs: usize) {
        let svc = engine.get_index(name).unwrap_or_else(|| panic!("[{name}] was removed"));
        assert_eq!(svc.uuid, local, "[{name}] was replaced by another local index");
        let shard = svc.open_shard(0, ShardOpenSpec { allocation_id: "aid".into(), primary: true, primary_term: 1, recovery: EngineRecovery::ExistingStore }).unwrap();
        assert_eq!(shard.shard.max_seq_no(), docs as i64 - 1, "documents lost");
    }

    /// Strip what the copy records about its incarnation in its metadata
    /// file (a copy written before the identity was persisted there).
    fn forget_persisted_identity(index_dir: &Path) {
        let p = index_dir.join("index_meta.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        v.as_object_mut().unwrap().remove("cluster_uuid");
        std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
    }

    #[test]
    fn missing_truncated_and_stale_maps_never_delete_an_intact_copy() {
        for variant in ["missing", "truncated", "garbage", "stale", "empty-object"] {
            for incremental in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let md = cluster_md("keep", "cluster.uuid.keep");
                let local = node_with_data(dir.path(), &md, "keep", 5, false);
                assert_ne!(local, "cluster.uuid.keep");
                let map = dir.path().join("uuids.json");
                let good = std::fs::read(&map).unwrap();
                match variant {
                    "missing" => std::fs::remove_file(&map).unwrap(),
                    "truncated" => std::fs::write(&map, &good[..good.len() / 2]).unwrap(),
                    "garbage" => std::fs::write(&map, b"\0\0\0\xff").unwrap(),
                    "stale" => std::fs::write(&map, serde_json::to_vec(&json!({"indices": {"keep": ["cluster.uuid.keep", "someOtherLocalUuid0000"]}})).unwrap()).unwrap(),
                    _ => std::fs::write(&map, b"{\"indices\":{}}").unwrap(),
                }
                let (engine, mut uuids) = start(dir.path());
                let rep = if incremental { apply_changes(&engine, &md, &md, &mut uuids, true, &Default::default()) } else { apply(&engine, &md, &mut uuids) };
                assert!(rep.deleted.is_empty() && rep.quarantined.is_empty(), "{variant}: {rep:?}");
                assert!(rep.errors.is_empty(), "{variant}: {:?}", rep.errors);
                assert_eq!(rep.reconstructed, vec!["keep".to_string()], "{variant}");
                assert_intact(&engine, "keep", &local, 5);
                // The map is repaired, durably.
                assert_eq!(UuidMap::load(&map).unwrap().indices["keep"], ("cluster.uuid.keep".to_string(), local.clone()), "{variant}");
                if matches!(variant, "truncated" | "garbage") {
                    let kept = std::fs::read_dir(dir.path()).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("uuids.corrupt-"));
                    assert!(kept, "{variant}: the corrupt map is kept for inspection");
                }
            }
        }
    }

    #[test]
    fn an_unreadable_map_fails_the_start_and_is_never_read_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("keep", "cluster.uuid.keep");
        let local = node_with_data(dir.path(), &md, "keep", 3, false);
        let map = dir.path().join("uuids.json");
        std::fs::remove_file(&map).unwrap();
        // Reading it fails with something other than "not found".
        std::fs::create_dir(&map).unwrap();
        let err = UuidMap::load(&map).unwrap_err();
        assert_ne!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(UuidMap::load_at_start(&map).is_err(), "an unreadable map must fail the start");
        assert!(map.is_dir(), "an unreadable map is not moved aside");
        // Repaired by the operator: the copy is still there.
        std::fs::remove_dir(&map).unwrap();
        let (engine, mut uuids) = start(dir.path());
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.deleted.is_empty(), "{rep:?}");
        assert_intact(&engine, "keep", &local, 3);
    }

    #[test]
    fn identity_is_reconstructed_from_the_copy_record_of_an_older_copy() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("keep", "cluster.uuid.keep");
        let local = node_with_data(dir.path(), &md, "keep", 4, true);
        {
            let (engine, _) = start(dir.path());
            forget_persisted_identity(engine.get_index("keep").unwrap().path());
        }
        std::fs::remove_file(dir.path().join("uuids.json")).unwrap();
        let (engine, mut uuids) = start(dir.path());
        assert_eq!(engine.get_index("keep").unwrap().cluster_uuid(), None);
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.deleted.is_empty() && rep.quarantined.is_empty(), "{rep:?}");
        assert_intact(&engine, "keep", &local, 4);
        // And the identity is now in the copy's own metadata file.
        let svc = engine.get_index("keep").unwrap();
        assert_eq!(elasticsearch_engine::index::IndexCore::persisted_cluster_uuid(svc.path()).as_deref(), Some("cluster.uuid.keep"));
    }

    #[test]
    fn a_copy_of_unknown_identity_is_quarantined_not_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("keep", "cluster.uuid.keep");
        let local = node_with_data(dir.path(), &md, "keep", 6, false);
        let old_path = {
            let (engine, _) = start(dir.path());
            let p = engine.get_index("keep").unwrap().path().to_path_buf();
            forget_persisted_identity(&p);
            p
        };
        std::fs::remove_file(dir.path().join("uuids.json")).unwrap();
        let (engine, mut uuids) = start(dir.path());
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.deleted.is_empty(), "{rep:?}");
        assert_eq!(rep.quarantined.len(), 1, "{rep:?}");
        let (_, kept) = &rep.quarantined[0];
        assert!(!old_path.exists());
        assert!(kept.join("shard-0").join("shard_state.json").exists(), "the shard data is kept whole");
        assert!(kept.starts_with(dir.path().join("engine").join(Engine::QUARANTINE_DIR)));
        // The name holds a fresh shell of the cluster's incarnation.
        let shell = engine.get_index("keep").unwrap();
        assert_ne!(shell.uuid, local);
        assert_eq!(shell.cluster_uuid(), Some("cluster.uuid.keep"));
        // Listed as dangling, under its local uuid (nothing else is known).
        let d = crate::rest::dangling::quarantined_copy(kept).unwrap();
        assert_eq!((d.name.as_str(), d.uuid.as_str()), ("keep", local.as_str()));
        // A restart neither reopens nor deletes it.
        drop((shell, engine));
        let (engine, mut uuids) = start(dir.path());
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.deleted.is_empty() && rep.quarantined.is_empty() && rep.errors.is_empty(), "{rep:?}");
        assert!(kept.join("shard-0").join("shard_state.json").exists());
    }

    #[test]
    fn a_metadata_only_shell_of_unknown_identity_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("shell", "cluster.uuid.shell");
        {
            let (engine, mut uuids) = start(dir.path());
            assert!(apply(&engine, &md, &mut uuids).errors.is_empty());
            forget_persisted_identity(engine.get_index("shell").unwrap().path());
        }
        std::fs::remove_file(dir.path().join("uuids.json")).unwrap();
        let (engine, mut uuids) = start(dir.path());
        let rep = apply(&engine, &md, &mut uuids);
        assert_eq!(rep.deleted, vec!["shell".to_string()], "{rep:?}");
        assert!(rep.quarantined.is_empty());
        assert!(engine.get_index("shell").is_some());
        assert!(!dir.path().join("engine").join(Engine::QUARANTINE_DIR).exists());
    }

    #[test]
    fn a_genuine_older_incarnation_is_still_replaced() {
        for tombstoned in [true, false] {
            for (evidence, map_lost) in [("metadata", true), ("copy-record", true), ("map", false)] {
                let dir = tempfile::tempdir().unwrap();
                let old = cluster_md("idx", "cluster.uuid.old");
                node_with_data(dir.path(), &old, "idx", 2, evidence == "copy-record");
                if evidence != "metadata" {
                    let (engine, _) = start(dir.path());
                    forget_persisted_identity(engine.get_index("idx").unwrap().path());
                }
                if map_lost {
                    std::fs::remove_file(dir.path().join("uuids.json")).unwrap();
                }
                let mut new = cluster_md("idx", "cluster.uuid.new");
                if tombstoned {
                    tombstone(&mut new, "idx", "cluster.uuid.old");
                }
                let (engine, mut uuids) = start(dir.path());
                let rep = apply(&engine, &new, &mut uuids);
                assert!(rep.quarantined.is_empty(), "{evidence}");
                // Only the map recorded it (a copy from before the identity was
                // persisted, never a started shard copy): nothing committed
                // depends on it.
                if tombstoned || evidence == "map" {
                    assert_eq!(rep.deleted, vec!["idx".to_string()], "{evidence}: {rep:?}");
                    assert!(rep.dangling.is_empty(), "{evidence}");
                } else {
                    // ES: no tombstone, so the old incarnation stays dangling.
                    assert!(rep.deleted.is_empty(), "{evidence}: {rep:?}");
                    assert_eq!(rep.dangling.len(), 1, "{evidence}: {rep:?}");
                    let kept = &rep.dangling[0].1;
                    assert!(kept.join("shard-0").join("shard_state.json").exists(), "{evidence}");
                    let d = crate::rest::dangling::quarantined_copy(kept).unwrap();
                    assert_eq!((d.name.as_str(), d.uuid.as_str()), ("idx", "cluster.uuid.old"), "{evidence}");
                }
                let svc = engine.get_index("idx").unwrap();
                assert_eq!(svc.cluster_uuid(), Some("cluster.uuid.new"), "{evidence}");
                assert!(!crate::mirror::holds_shard_data(svc.path()), "{evidence}");
            }
        }
    }

    fn tombstone(md: &mut Metadata, name: &str, uuid: &str) {
        let key = elasticsearch_cluster::graveyard::CUSTOM;
        let mut stones = md.customs.get(key).and_then(|g| g["tombstones"].as_array().cloned()).unwrap_or_default();
        stones.push(json!({"index": {"index_name": name, "index_uuid": uuid}, "delete_date_in_millis": 1}));
        md.customs.insert(key.into(), json!({"tombstones": stones}));
    }

    /// A local index the metadata does not hold: deleted when a tombstone
    /// names it, else kept dangling (out of service, listed, its data whole);
    /// an import (the index back in the metadata under its uuid) re-registers
    /// that very copy, and a later tombstone deletes a dangling copy.
    #[test]
    fn absent_indices_are_deleted_only_when_tombstoned() {
        for path in ["full", "incremental"] {
            // Tombstoned: deleted.
            let dir = tempfile::tempdir().unwrap();
            let md = cluster_md("gone", "cluster.uuid.gone");
            let local = node_with_data(dir.path(), &md, "gone", 3, true);
            let mut after = Metadata::default();
            tombstone(&mut after, "gone", "cluster.uuid.gone");
            let (engine, mut uuids) = start(dir.path());
            let rep = if path == "full" { apply(&engine, &after, &mut uuids) } else { apply_changes(&engine, &md, &after, &mut uuids, false, &Default::default()) };
            assert_eq!(rep.deleted, vec!["gone".to_string()], "{path}: {rep:?}");
            assert!(rep.dangling.is_empty() && engine.get_index("gone").is_none(), "{path}");
            assert!(crate::rest::dangling::quarantined_in(&engine.data).is_empty(), "{path}");
            let _ = local;

            // Not tombstoned: dangling, then imported back intact.
            let dir = tempfile::tempdir().unwrap();
            let md = cluster_md("keep", "cluster.uuid.keep");
            let local = node_with_data(dir.path(), &md, "keep", 4, true);
            let (engine, mut uuids) = start(dir.path());
            let empty = Metadata::default();
            let rep = if path == "full" { apply(&engine, &empty, &mut uuids) } else { apply_changes(&engine, &md, &empty, &mut uuids, false, &Default::default()) };
            assert!(rep.deleted.is_empty(), "{path}: {rep:?}");
            assert_eq!(rep.dangling.len(), 1, "{path}: {rep:?}");
            assert!(engine.get_index("keep").is_none(), "{path}: out of service");
            let listed = crate::rest::dangling::quarantined_in(&engine.data);
            assert_eq!(listed.len(), 1, "{path}");
            assert_eq!((listed[0].0.name.as_str(), listed[0].0.uuid.as_str()), ("keep", "cluster.uuid.keep"), "{path}");
            // The master never publishes it as a new index.
            assert!(!extract(&engine, &empty, &mut uuids).indices.contains_key("keep"), "{path}");
            // Imported: the same copy, every document.
            let rep = apply_changes(&engine, &empty, &md, &mut uuids, false, &Default::default());
            assert!(rep.errors.is_empty(), "{path}: {:?}", rep.errors);
            assert_eq!(rep.created, vec!["keep".to_string()], "{path}");
            assert_intact(&engine, "keep", &local, 4);
            assert!(crate::rest::dangling::quarantined_in(&engine.data).is_empty(), "{path}");

            // Dangling again, then tombstoned: the set-aside copy is deleted.
            let rep = apply_changes(&engine, &md, &empty, &mut uuids, false, &Default::default());
            assert_eq!(rep.dangling.len(), 1, "{path}");
            let kept = rep.dangling[0].1.clone();
            let mut later = Metadata::default();
            tombstone(&mut later, "keep", "cluster.uuid.keep");
            let rep = if path == "full" { apply(&engine, &later, &mut uuids) } else { apply_changes(&engine, &empty, &later, &mut uuids, false, &Default::default()) };
            assert!(rep.errors.is_empty(), "{path}: {:?}", rep.errors);
            assert!(!kept.exists(), "{path}: a tombstoned dangling copy is deleted");
        }
    }

    /// An index the metadata never committed here (a rolled-back creation)
    /// or that holds no shard data is simply deleted.
    #[test]
    fn uncommitted_or_dataless_absent_indices_are_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let md = cluster_md("shell", "cluster.uuid.shell");
        let mut uuids = UuidMap::default();
        assert!(apply(&engine, &md, &mut uuids).errors.is_empty());
        let svc = engine.create_index("rolled-back", &json!({"settings": {"number_of_shards": 1}})).unwrap();
        assert!(holds_shard_data(svc.path()));
        drop(svc);
        let rep = apply(&engine, &Metadata::default(), &mut uuids);
        let mut deleted = rep.deleted.clone();
        deleted.sort();
        assert_eq!(deleted, vec!["rolled-back".to_string(), "shell".to_string()], "{rep:?}");
        assert!(rep.dangling.is_empty());
    }

    #[test]
    fn conflicting_identity_evidence_quarantines() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("keep", "cluster.uuid.keep");
        node_with_data(dir.path(), &md, "keep", 2, false);
        // The copy record names another incarnation than its metadata file.
        {
            let (engine, _) = start(dir.path());
            let svc = engine.get_index("keep").unwrap();
            crate::shards::write_copy_record(&svc, &crate::shards::CopyRecord { index: "keep".into(), index_uuid: "cluster.uuid.other".into(), shard: 0,
                allocation_id: "aid".into(), primary_term: 1, primary: true, history_uuid: None, global_checkpoint: None }).unwrap();
        }
        // A consistent map entry is trusted as is (no per-application file
        // reads); without it the copy's own evidence is weighed.
        std::fs::remove_file(dir.path().join("uuids.json")).unwrap();
        let (engine, mut uuids) = start(dir.path());
        assert!(matches!(incarnation(&engine.get_index("keep").unwrap(), "cluster.uuid.keep", &uuids), Incarnation::Unknown(_)));
        let rep = apply(&engine, &md, &mut uuids);
        assert!(rep.deleted.is_empty(), "{rep:?}");
        assert_eq!(rep.quarantined.len(), 1, "{rep:?}");
    }

    #[test]
    fn uuid_map_saves_are_durable_and_failures_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let md = cluster_md("keep", "cluster.uuid.keep");
        let (engine, mut uuids) = start(dir.path());
        assert!(apply(&engine, &md, &mut uuids).errors.is_empty());
        let map = dir.path().join("uuids.json");
        let saved = std::fs::read(&map).unwrap();
        assert!(!dir.path().join("uuids.tmp").exists());
        save_test_hook::fail_under(dir.path(), true);
        let mut changed = md.clone();
        changed.indices.insert("more".into(), { let mut m = cluster_md("more", "cluster.uuid.more").indices["more"].clone(); m.name = "more".into(); m });
        let rep = apply(&engine, &changed, &mut uuids);
        save_test_hook::fail_under(dir.path(), false);
        assert!(rep.errors.iter().any(|(k, _)| k == UUIDS_FAILURE), "{:?}", rep.errors);
        assert_eq!(std::fs::read(&map).unwrap(), saved, "the previous map stays whole");
        // Retried by the next application.
        let rep = apply(&engine, &changed, &mut uuids);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        assert!(UuidMap::load(&map).unwrap().indices.contains_key("more"));
    }
}
