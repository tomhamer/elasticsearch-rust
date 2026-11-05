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
    pub fn load(path: &Path) -> UuidMap {
        let mut m: UuidMap = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        m.path = Some(path.to_path_buf());
        m
    }

    pub(crate) fn save(&self) -> std::io::Result<()> {
        let Some(p) = &self.path else { return Ok(()) };
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(tmp, p)
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

/// Build cluster metadata from the master's engine, carrying over the
/// cluster-owned parts of `prev` (uuids, primary terms, in-sync ids,
/// customs the engine does not know about) and bumping per-index versions
/// where something changed.
pub fn extract(engine: &Engine, prev: &Metadata, uuids: &mut UuidMap) -> Metadata {
    let mut md = prev.clone();
    md.indices.clear();
    for svc in engine.all_indices() {
        let im = extract_index(engine, &svc, prev, uuids);
        md.indices.insert(svc.name.clone(), im);
    }
    uuids.indices.retain(|n, _| md.indices.contains_key(n));
    let _ = uuids.save();
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
    let old = prev.indices.get(&svc.name).filter(|_| uuids.indices.get(&svc.name).map(|(_, l)| l == &svc.uuid).unwrap_or(true));
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
    // Deleted indices, and indices recreated under the same name.
    for svc in engine.all_indices() {
        let keep = match (md.indices.get(&svc.name), uuids.indices.get(&svc.name)) {
            (Some(im), Some((cluster, local))) => cluster == &im.uuid && local == &svc.uuid,
            (Some(_), None) => false,
            (None, _) => false,
        };
        if !keep {
            remove_index(engine, &svc.name, uuids, &mut rep);
        }
    }
    for (name, im) in &md.indices {
        apply_index(engine, name, im, uuids, &mut rep);
    }
    apply_globals(engine, md);
    if let Err(e) = engine.persist_state() {
        rep.errors.push(("_state".into(), e.reason.clone()));
    }
    let _ = uuids.save();
    rep
}

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
    for (name, p) in &prev.indices {
        let replaced = md.indices.get(name).is_none_or(|im| im.uuid != p.uuid);
        if replaced && engine.get_index(name).is_some() {
            remove_index(engine, name, uuids, &mut rep);
        }
    }
    for (name, im) in &md.indices {
        let mapped = uuids.indices.get(name).is_some_and(|(cluster, _)| cluster == &im.uuid);
        let same = prev.indices.get(name).is_some_and(|p| same_engine_view(p, im));
        // `retry`: indices whose last application failed are applied again.
        if !same || !mapped || retry.contains(name) || engine.get_index(name).is_none() {
            if !mapped && engine.get_index(name).is_some() {
                // Another incarnation under this name.
                remove_index(engine, name, uuids, &mut rep);
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
    if globals || !rep.created.is_empty() || !rep.deleted.is_empty() || !rep.updated.is_empty() {
        if let Err(e) = engine.persist_state() {
            rep.errors.push(("_state".into(), e.reason.clone()));
        }
        let _ = uuids.save();
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
        None => match create_sparse(engine, name, im) {
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
    // The copy's incarnation, readable without the uuid maps.
    if uuids.indices.get(name).is_some_and(|(cluster, local)| cluster == &im.uuid && local == &svc.uuid) {
        svc.set_cluster_uuid(&im.uuid);
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
        let needs_close = svc.is_open() && delta.keys().any(|k| !es_settings::is_dynamic(k));
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
}
