//! Durable node-local state under `<path.data>/_cluster` (ES's
//! `PersistedClusterStateService` + `NodeEnvironment` node id).
//!
//! * `node_id` — generated once, stable across restarts (operators and the
//!   Kubernetes tooling key shutdown records and exclusions on it).
//! * `state-<generation>.json` — the coordinator's [`PersistedState`]
//!   (current term + last accepted cluster state), written incrementally
//!   as ES's `PersistedClusterStateService` does (one document per index,
//!   one for the global metadata, rewritten only when they change). Each
//!   generation is one file: the small rest of the state (nodes, blocks,
//!   versions), plus every index entry and the global metadata, each
//!   either inline (changed since the generation it was last written in)
//!   or a reference to the earlier generation holding it. A publication
//!   that changes one index writes that index and the small rest: one
//!   file, written temp → fsync → rename → fsync(dir), so a crash leaves
//!   the previous generation or the new one, never a torn one. The
//!   coordinator's ordering contract requires this to be durable before
//!   any later message is sent. Format 2 checksums its exact JSON
//!   payload; format 3 additionally requires mirrored-generation recovery. The previous root and its dependencies remain available for
//!   corruption fallback inspection. Each generation also has an independent
//!   checked recovery copy, written first; same-generation recovery is automatic.
//!   Startup is fenced if fallback would
//!   forget a newer term or accepted state; authoritative recovery is required.
//!   Older unreferenced generations are deleted after
//!   the new one is durable. Once more than
//!   [`MAX_REFERENCED`] older generations are referenced, the next one is
//!   written in full (compaction).
//! * `coordination.json` — the previous format (the whole state in one
//!   file, rewritten on every publication). Read once on upgrade, then
//!   replaced by a marker an older version refuses to start on (a
//!   downgrade would otherwise read a stale term and state).
//!
//! JSON (the human-readable form of the state; its binary wire form is
//! `elasticsearch_cluster::wire`): readable by operators.

use elasticsearch_cluster::coordination::{PersistUpdate, PersistedState};
use elasticsearch_cluster::state::{random_id, ClusterState, IndexMetadata, IndexRoutingTable, Metadata, RoutingTable};
use serde::{Deserialize, Serialize};
use elasticsearch_engine::rustic_core::codec::crc32;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The incremental format's version.
const FORMAT: u32 = 3;
const CHECKED_FORMAT: u32 = 2;
const LEGACY_FORMAT: u32 = 1;
const STATE_PREFIX: &str = "state-";
const RECOVERY_PREFIX: &str = "recovery-state-";
const LEGACY: &str = "coordination.json";
/// Older generations a generation may reference before the next one is
/// written in full.
pub const MAX_REFERENCED: usize = 16;

pub struct Gateway {
    dir: PathBuf,
    persisted: PersistedState,
    disk: Disk,
}

/// A last-good snapshot was found, but newer durable coordination records
/// are unreadable. Returning a running Gateway here could lower a persisted
/// term or forget an accepted value and allow an unsafe second vote.
///
/// Recovery tools can downcast `io::Error::get_ref()` to inspect this snapshot.
/// It is NOT authorized for coordinator startup: resynchronize/recover from an
/// authoritative cluster copy before restarting this node. No files are changed
/// by this fallback path; the failed generations remain available for diagnosis.
#[derive(Debug)]
pub struct GatewayRecoveryRequired {
    pub newest_generation: u64,
    pub recovered_generation: u64,
    last_good_state: PersistedState,
    failures: Vec<String>,
}

impl GatewayRecoveryRequired {
    pub fn last_good_state(&self) -> &PersistedState {
        &self.last_good_state
    }
}

impl std::fmt::Display for GatewayRecoveryRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gateway corruption after generation [{}]: last good generation [{}] is available for recovery, but coordinator startup is fenced because newer term/accepted state may be lost; authoritative recovery is required: {}",
            self.newest_generation, self.recovered_generation, self.failures.join("; "))
    }
}
impl std::error::Error for GatewayRecoveryRequired {}

/// Where each part of the persisted state lives on disk.
#[derive(Default)]
struct Disk {
    /// The latest durable generation (0: none yet).
    generation: u64,
    /// The generation holding each index's entry.
    indices: BTreeMap<String, u64>,
    /// The generation holding the global metadata.
    global: u64,
    /// `coordination.json` is the marker (or absent): nothing to replace.
    legacy_replaced: bool,
    /// Rewrite all entries on the first write after an unchecked-format load.
    upgrade: bool,
}

/// One part of a generation: written here, or in an earlier generation.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Entry<T> {
    Inline(T),
    In(u64),
}

/// The global metadata (ES's global metadata document).
#[derive(Serialize)]
struct GlobalsRef<'a> {
    persistent_settings: &'a Map<String, Value>,
    transient_settings: &'a Map<String, Value>,
    index_templates: &'a BTreeMap<String, Value>,
    component_templates: &'a BTreeMap<String, Value>,
    legacy_templates: &'a BTreeMap<String, Value>,
    customs: &'a BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct Globals {
    persistent_settings: Map<String, Value>,
    transient_settings: Map<String, Value>,
    index_templates: BTreeMap<String, Value>,
    component_templates: BTreeMap<String, Value>,
    legacy_templates: BTreeMap<String, Value>,
    customs: BTreeMap<String, Value>,
}

/// One index: its metadata and routing table (ES keeps the routing in the
/// cluster state document; here it rides with its index, so a shard
/// starting rewrites only its own index).
#[derive(Serialize)]
struct IndexRef<'a> {
    metadata: Option<&'a IndexMetadata>,
    routing: Option<&'a IndexRoutingTable>,
}

#[derive(Deserialize)]
struct IndexEntry {
    metadata: Option<IndexMetadata>,
    routing: Option<IndexRoutingTable>,
}

#[derive(Serialize)]
struct GenerationOut<'a> {
    format: u32,
    generation: u64,
    current_term: u64,
    /// The state without its indices, routing table entries and global
    /// metadata maps.
    state: &'a ClusterState,
    global: Entry<GlobalsRef<'a>>,
    indices: BTreeMap<&'a str, Entry<IndexRef<'a>>>,
}

#[derive(Deserialize)]
struct GenerationIn {
    format: u32,
    generation: u64,
    current_term: u64,
    state: ClusterState,
    global: Entry<Globals>,
    indices: BTreeMap<String, Entry<IndexEntry>>,
}

/// Written to `state-N` before the first mirrored generation's recovery copy,
/// so a reader that ignores recovery files refuses to start (see
/// `write_generation`). Until `recovery-state-N` exists, generation N was never
/// durable or acknowledged: it is not a gateway generation.
const RECOVERY_PLACEHOLDER: &[u8] = br#"{"format":3,"state":"this gateway generation was written by a newer version (format 3); downgrading this node is not supported","recovery_pending":true}"#;

/// What a pre-checksum build (master, gateway format 1) reports when it meets
/// a checked generation. It deserializes `state` as a cluster state, so this
/// string turns its "missing field" into an explicit refusal. It cannot be
/// changed, and ES also never lets a node start on data a newer version wrote
/// (`NodeMetadata.verifyUpgradeToCurrentVersion`): rollback is a restore.
const DOWNGRADE_NOTICE: &str = "this gateway generation was written by a newer version (format 3); downgrading this node is not supported";

/// A downgrade fence left by a crash before generation N's first copy.
fn unwritten_placeholder(dir: &Path, generation: u64) -> bool {
    !recovery_path(dir, generation).exists()
        && fs::read(generation_path(dir, generation)).ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|v| v == serde_json::from_slice::<Value>(RECOVERY_PLACEHOLDER).expect("placeholder is JSON"))
}

/// The durable generations: present in either copy, minus placeholders whose
/// write never reached its recovery copy.
fn written_generations(dir: &Path) -> std::io::Result<Vec<u64>> {
    Ok(generations(dir)?.into_iter().filter(|g| !unwritten_placeholder(dir, *g)).collect())
}

fn write_durably(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent() {
        // fsync the directory so the rename itself is durable.
        elasticsearch_engine::rustic_core::platform::sync_dir(parent)?;
    }
    Ok(())
}

/// `<data>/_cluster/segment_format.json`: the cluster's segment format floor
/// as this node last applied it from a *committed* cluster state, with that
/// cluster's uuid. A restarting node writes in this floor until a master
/// publishes, never in a floor it only accepted: an uncommitted raise may be
/// superseded by a new leader, and a node that applied it would refuse every
/// later publication and stay out of the cluster. The last accepted state
/// (which may carry such a raise) is deliberately not consulted. Losing an
/// update (a crash before it is written) only makes the node start from an
/// older committed floor, which every member still reads.
const SEGMENT_FORMAT_FILE: &str = "segment_format.json";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommittedSegmentFormat {
    cluster_uuid: String,
    floor: u32,
}

fn read_committed_segment_format(dir: &Path) -> std::io::Result<Option<(String, u32)>> {
    let path = dir.join(SEGMENT_FORMAT_FILE);
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<CommittedSegmentFormat>(&bytes)
            .map(|m| Some((m.cluster_uuid, m.floor)))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("corrupt {}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// The committed segment format floor this node last applied (see
/// [`SEGMENT_FORMAT_FILE`]): `(cluster uuid, floor)`, `None` if it never
/// applied one.
pub fn load_committed_segment_format(data: &Path) -> std::io::Result<Option<(String, u32)>> {
    read_committed_segment_format(&data.join("_cluster"))
}

/// Durably record the floor of a committed state this node applied.
pub fn store_committed_segment_format(data: &Path, cluster_uuid: &str, floor: u32) -> std::io::Result<()> {
    let dir = data.join("_cluster");
    fs::create_dir_all(&dir)?;
    let bytes = serde_json::to_vec(&CommittedSegmentFormat { cluster_uuid: cluster_uuid.to_string(), floor })
        .map_err(std::io::Error::other)?;
    write_durably(&dir.join(SEGMENT_FORMAT_FILE), &bytes)
}

/// ES's `NodeEnvironment` node lock: `<data>/node.lock`, held (flock) for the
/// life of the process. Two processes on one data path (a restart while the
/// previous process is still exiting) would interleave gateway generations
/// and shard files. Waits up to `wait` for a previous holder to exit, then
/// fails the start. A path this process already holds (in-process test
/// nodes) is not locked twice.
pub fn lock_data_path(data: &Path, wait: std::time::Duration) -> std::io::Result<()> {
    use std::collections::HashMap;
    static HELD: std::sync::Mutex<Option<HashMap<PathBuf, File>>> = std::sync::Mutex::new(None);
    let path = fs::canonicalize(data)?.join("node.lock");
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let held = held.get_or_insert_with(HashMap::new);
    if held.contains_key(&path) {
        return Ok(());
    }
    let file = fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path)?;
    let deadline = std::time::Instant::now() + wait;
    let mut logged = false;
    loop {
        if try_lock_exclusive(&file)? {
            held.insert(path, file);
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, format!(
                "failed to obtain node lock [{}]: is another node running on path.data [{}]?", path.display(), data.display())));
        }
        if !logged {
            tracing::warn!(path = %path.display(), "node lock held by another process (a previous node still exiting?); waiting up to {wait:?}");
            logged = true;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[cfg(unix)]
fn try_lock_exclusive(f: &File) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    // SAFETY: flock on a file descriptor this function borrows.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    if e.kind() == std::io::ErrorKind::WouldBlock { Ok(false) } else { Err(e) }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_f: &File) -> std::io::Result<bool> {
    Ok(true)
}

/// The node id stored in `<data>/_cluster/node_id`, created on first start.
pub fn load_or_create_node_id(data: &Path) -> std::io::Result<String> {
    let dir = data.join("_cluster");
    fs::create_dir_all(&dir)?;
    let p = dir.join("node_id");
    if let Ok(s) = fs::read_to_string(&p) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return Ok(s);
        }
    }
    let id = random_id();
    write_durably(&p, id.as_bytes())?;
    Ok(id)
}

fn generation_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("{STATE_PREFIX}{generation}.json"))
}

fn recovery_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("{RECOVERY_PREFIX}{generation}.json"))
}

/// The generations present in either durable copy, ascending.
fn generations(dir: &Path) -> std::io::Result<Vec<u64>> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir)? {
        let name = e?.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(g) = name.strip_prefix(STATE_PREFIX).or_else(|| name.strip_prefix(RECOVERY_PREFIX)).and_then(|n| n.strip_suffix(".json")).and_then(|n| n.parse::<u64>().ok()) {
            out.push(g);
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The checksum covers the exact JSON payload bytes, not a reserialized map.
/// RawValue keeps operator-readable JSON and rejects truncated envelopes.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedGeneration {
    format: u32,
    /// [`DOWNGRADE_NOTICE`]: outside the checksum and ignored on read.
    #[serde(default, rename = "state", skip_serializing_if = "Option::is_none")]
    downgrade_notice: Option<String>,
    crc32: u32,
    payload: Box<serde_json::value::RawValue>,
}

fn checked_bytes(payload: Vec<u8>) -> std::io::Result<Vec<u8>> {
    let checksum = crc32(&payload);
    let payload = serde_json::value::RawValue::from_string(
        String::from_utf8(payload).map_err(std::io::Error::other)?,
    ).map_err(std::io::Error::other)?;
    serde_json::to_vec(&CheckedGeneration { format: FORMAT, downgrade_notice: Some(DOWNGRADE_NOTICE.into()), crc32: checksum, payload })
        .map_err(std::io::Error::other)
}

fn read_generation(dir: &Path, generation: u64) -> std::io::Result<GenerationIn> {
    match read_generation_file(&generation_path(dir, generation), generation) {
        Ok(state) => Ok(state),
        Err(error) if matches!(error.kind(), std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied) => Err(error),
        Err(primary) => {
            // Only the SAME durable generation can replace a damaged copy.
            // Falling back to an older state could forget a term or accepted
            // value and permit a second vote. Never repair/overwrite evidence.
            let backup = read_generation_file(&recovery_path(dir, generation), generation)
                .map_err(|backup| if matches!(backup.kind(), std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied) {
                    backup
                } else { std::io::Error::other(format!("{primary}; recovery copy: {backup}")) })?;
            if backup.format != FORMAT {
                return Err(std::io::Error::other("recovery copy must be checksummed"));
            }
            tracing::warn!(generation, error = %primary, "using checked gateway recovery generation");
            Ok(backup)
        }
    }
}

fn read_generation_file(p: &Path, generation: u64) -> std::io::Result<GenerationIn> {
    let bytes = fs::read(&p)?;
    let corrupt = |reason: String| std::io::Error::new(std::io::ErrorKind::InvalidData,
        format!("corrupt {}: {reason}", p.display()));
    #[derive(Deserialize)]
    struct Header { format: u32 }
    let header: Header = serde_json::from_slice(&bytes).map_err(|e| corrupt(e.to_string()))?;
    let g: GenerationIn = match header.format {
        LEGACY_FORMAT => serde_json::from_slice(&bytes).map_err(|e| corrupt(e.to_string()))?,
        CHECKED_FORMAT | FORMAT => {
            let checked: CheckedGeneration = serde_json::from_slice(&bytes).map_err(|e| corrupt(e.to_string()))?;
            if crc32(checked.payload.get().as_bytes()) != checked.crc32 {
                return Err(corrupt("cluster state checksum mismatch".into()));
            }
            let g: GenerationIn = serde_json::from_str(checked.payload.get()).map_err(|e| corrupt(e.to_string()))?;
            if g.format != header.format { return Err(corrupt("payload format mismatch".into())); }
            g
        }
        version => {
            // The envelope's format is outside the checksum. Refuse a downgrade
            // only for a copy that verifiably IS a newer format: a checksummed
            // payload that names the same format. Anything else (one flipped
            // envelope bit, a truncated write) is corruption, so the recovery
            // copy and older generations are still consulted.
            #[derive(Deserialize)]
            struct Envelope { crc32: u32, payload: Box<serde_json::value::RawValue> }
            let verified = serde_json::from_slice::<Envelope>(&bytes).ok()
                .filter(|e| crc32(e.payload.get().as_bytes()) == e.crc32)
                .and_then(|e| serde_json::from_str::<Value>(e.payload.get()).ok())
                .is_some_and(|payload| payload["format"].as_u64() == Some(u64::from(version)));
            if verified {
                return Err(std::io::Error::new(std::io::ErrorKind::Unsupported,
                    format!("{}: unsupported gateway format [{version}]; refusing a downgrade", p.display())));
            }
            return Err(corrupt(format!("unrecognized gateway format [{version}] without a matching checksummed payload (corruption, or an unverifiable newer format)")));
        }
    };
    if g.generation != generation {
        return Err(corrupt(format!("generation [{}] does not match filename [{generation}]", g.generation)));
    }
    Ok(g)
}

/// The cluster settings of a state read from disk, as ES's gateway leaves
/// them. Transient settings are dropped: ES never persists them, so they do
/// not survive a full cluster restart (earlier versions of this server
/// wrote them; a node rejoining a live cluster gets the master's full
/// state, transient settings included, as any joining node). Persistent
/// ones this version does not know, or whose value no longer validates,
/// are archived (`upgradeAndArchiveUnknownOrInvalidSettings`), never
/// failing the start or being applied.
fn settings_as_loaded(metadata: &mut Metadata) -> std::io::Result<bool> {
    let (transient, persistent) = (!metadata.transient_settings.is_empty(), metadata.persistent_settings.clone());
    if !metadata.transient_settings.is_empty() {
        tracing::info!("dropping {} transient cluster setting(s) written by an earlier start: transient settings do not survive a restart", metadata.transient_settings.len());
        metadata.transient_settings = Map::new();
    }
    let lists_changed = !elasticsearch_engine::settings::upgrade_legacy_setting_lists(&mut metadata.persistent_settings)
        .map_err(|e| std::io::Error::other(e.reason))?.is_empty();
    elasticsearch_engine::settings::archive_unknown_or_invalid_settings("persistent", &mut metadata.persistent_settings);
    let mut indices_changed = false;
    for index in metadata.indices.values_mut() {
        indices_changed |= !elasticsearch_engine::settings::upgrade_legacy_setting_lists(&mut index.settings)
            .map_err(|e| std::io::Error::other(format!("index [{}]: {}", index.name, e.reason)))?.is_empty();
        indices_changed |= !elasticsearch_engine::settings::archive_invalid_index_settings(&index.name, &mut index.settings).is_empty();
    }
    Ok(transient || persistent != metadata.persistent_settings || indices_changed || lists_changed)
}

/// The persisted state at the latest generation, with where each part lives.
fn load_generations(dir: &Path, latest: u64) -> std::io::Result<(PersistedState, Disk)> {
    let mut cache: HashMap<u64, GenerationIn> = HashMap::new();
    cache.insert(latest, read_generation(dir, latest)?);
    // Every entry the latest generation names, by the generation holding it.
    let (global_at, index_at): (u64, Vec<(String, u64)>) = {
        let top = &cache[&latest];
        let at = |e: &Entry<IndexEntry>| match e {
            Entry::Inline(_) => latest,
            Entry::In(g) => *g,
        };
        let global_at = match &top.global {
            Entry::Inline(_) => latest,
            Entry::In(g) => *g,
        };
        (global_at, top.indices.iter().map(|(n, e)| (n.clone(), at(e))).collect())
    };
    for g in std::iter::once(global_at).chain(index_at.iter().map(|(_, g)| *g)) {
        if g > latest {
            return Err(std::io::Error::other(format!("generation [{latest}] references a later generation [{g}]")));
        }
        if let std::collections::hash_map::Entry::Vacant(v) = cache.entry(g) {
            v.insert(read_generation(dir, g)?);
        }
    }
    let missing = |what: &str, g: u64| std::io::Error::other(format!("generation [{latest}]: {what} is not inline in generation [{g}]"));
    let globals = match std::mem::replace(&mut cache.get_mut(&global_at).expect("loaded").global, Entry::In(0)) {
        Entry::Inline(globals) => globals,
        Entry::In(_) => return Err(missing("the global metadata", global_at)),
    };
    let mut entries = Vec::with_capacity(index_at.len());
    for (name, g) in &index_at {
        match cache.get_mut(g).expect("loaded").indices.remove(name) {
            Some(Entry::Inline(e)) => entries.push((name.clone(), e)),
            _ => return Err(missing(&format!("index [{name}]"), *g)),
        }
    }
    let upgrade = cache.values().any(|g| g.format != FORMAT);
    let top = cache.remove(&latest).expect("loaded");
    let mut state = top.state;
    state.metadata.persistent_settings = globals.persistent_settings;
    state.metadata.transient_settings = globals.transient_settings;
    state.metadata.index_templates = globals.index_templates;
    state.metadata.component_templates = globals.component_templates;
    state.metadata.legacy_templates = globals.legacy_templates;
    state.metadata.customs = globals.customs;
    for (name, e) in entries {
        if let Some(im) = e.metadata {
            state.metadata.indices.insert(name.clone(), im);
        }
        if let Some(r) = e.routing {
            state.routing_table.indices.insert(name, r);
        }
    }
    let disk = Disk { generation: latest, indices: index_at.into_iter().collect(), global: global_at, legacy_replaced: false, upgrade };
    Ok((PersistedState { current_term: top.current_term, last_accepted_state: state }, disk))
}

/// `coordination.json` after the upgrade: not a [`PersistedState`], so an
/// older version refuses to start instead of reading a stale state.
fn legacy_marker() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "format": "incremental",
        "note": "the cluster state is in state-<generation>.json, written by a newer version; downgrades are not supported",
    }))
    .expect("marker serializes")
}

/// Write generation `disk.generation + 1`: the parts of `new` that differ
/// from `old` (the state of the latest generation) inline, the rest as
/// references; retain the previous root and both roots' dependencies.
fn write_generation(dir: &Path, disk: &mut Disk, old: &ClusterState, term: u64, new: &ClusterState) -> std::io::Result<()> {
    // After fallback, never overwrite a rejected newer generation. Retain the
    // previous root AND its dependencies, including across full compaction.
    let previous: BTreeSet<u64> = disk.indices.values().copied()
        .chain([disk.global, disk.generation]).filter(|g| *g != 0).collect();
    let generation = generations(dir)?.last().copied().unwrap_or(0).max(disk.generation)
        .checked_add(1).ok_or_else(|| std::io::Error::other("gateway generation exhausted"))?;
    let referenced: BTreeSet<u64> = disk.indices.values().copied().chain((disk.global != 0).then_some(disk.global)).collect();
    // Compaction: rewrite everything once too many generations are live.
    let full = disk.generation == 0 || disk.upgrade || referenced.len() > MAX_REFERENCED;
    let om = &old.metadata;
    let nm = &new.metadata;
    let globals_same = om.persistent_settings == nm.persistent_settings
        && om.transient_settings == nm.transient_settings
        && om.index_templates == nm.index_templates
        && om.component_templates == nm.component_templates
        && om.legacy_templates == nm.legacy_templates
        && om.customs == nm.customs;
    let global = if !full && globals_same && disk.global != 0 {
        Entry::In(disk.global)
    } else {
        Entry::Inline(GlobalsRef {
            persistent_settings: &nm.persistent_settings,
            transient_settings: &nm.transient_settings,
            index_templates: &nm.index_templates,
            component_templates: &nm.component_templates,
            legacy_templates: &nm.legacy_templates,
            customs: &nm.customs,
        })
    };
    let names: BTreeSet<&str> = nm.indices.keys().chain(new.routing_table.indices.keys()).map(String::as_str).collect();
    let mut indices = BTreeMap::new();
    let mut locations = BTreeMap::new();
    for name in names {
        let (metadata, routing) = (nm.indices.get(name), new.routing_table.indices.get(name));
        let same = om.indices.get(name) == metadata && old.routing_table.indices.get(name) == routing;
        let entry = match disk.indices.get(name) {
            Some(&at) if !full && same => Entry::In(at),
            _ => Entry::Inline(IndexRef { metadata, routing }),
        };
        locations.insert(name.to_string(), match entry {
            Entry::In(at) => at,
            Entry::Inline(_) => generation,
        });
        indices.insert(name, entry);
    }
    // Every metadata field is in the skeleton or in the entries: a field
    // added to `Metadata` does not compile here until it is placed.
    let Metadata { cluster_uuid: _, cluster_uuid_committed: _, version: _, persistent_settings: _, transient_settings: _, indices: _, index_templates: _, component_templates: _, legacy_templates: _, customs: _ } = nm;
    let skeleton = ClusterState {
        cluster_name: new.cluster_name.clone(),
        version: new.version,
        state_uuid: new.state_uuid.clone(),
        master_node_id: new.master_node_id.clone(),
        nodes: new.nodes.clone(),
        coordination: new.coordination.clone(),
        metadata: Metadata { cluster_uuid: nm.cluster_uuid.clone(), cluster_uuid_committed: nm.cluster_uuid_committed, version: nm.version, ..Default::default() },
        routing_table: RoutingTable { version: new.routing_table.version, indices: BTreeMap::new() },
        blocks: new.blocks.clone(),
    };
    let global_at = match &global {
        Entry::In(at) => *at,
        Entry::Inline(_) => generation,
    };
    let out = GenerationOut { format: FORMAT, generation, current_term: term, state: &skeleton, global, indices };
    let bytes = serde_json::to_vec(&out).map_err(std::io::Error::other)?;
    let bytes = checked_bytes(bytes)?;
    // Persist the independent copy first. Once the primary can exist, the
    // matching checked copy is already durable. A crash between these writes
    // is safe: load enumerates both names and retains the newer accepted state.
    // Nothing is acknowledged before both are durable.
    if disk.generation == 0 || disk.upgrade {
        // Before the first mirrored-format recovery copy, fence old readers that
        // ignore recovery-state files and might otherwise reopen an older state.
        // A crash before the mirror is durable means this generation was never
        // written: a newer reader ignores the placeholder and loads N-1.
        write_durably(&generation_path(dir, generation), RECOVERY_PLACEHOLDER)?;
    }
    write_durably(&recovery_path(dir, generation), &bytes)?;
    write_durably(&generation_path(dir, generation), &bytes)?;
    disk.generation = generation;
    disk.indices = locations;
    disk.global = global_at;
    disk.upgrade = false;
    if !disk.legacy_replaced {
        // Always (a node that never had the old file too): an older version
        // then refuses to start instead of starting empty under this node's
        // id (term 0: a second vote in a term, or a second bootstrap).
        write_durably(&dir.join(LEGACY), &legacy_marker())?;
        disk.legacy_replaced = true;
    }
    // Generations the new one no longer needs (best effort: a leftover is
    // ignored on load and deleted next time).
    let live: BTreeSet<u64> = disk.indices.values().copied().chain([disk.global, generation]).chain(previous).collect();
    if let Ok(all) = generations(dir) {
        for g in all.into_iter().filter(|g| !live.contains(g)) {
            // A placeholder that never got its copies is not evidence of anything.
            if unwritten_placeholder(dir, g) {
                let _ = fs::remove_file(generation_path(dir, g));
                continue;
            }
            // Leave corrupt/unknown files as evidence for the operator.
            if read_generation_file(&generation_path(dir, g), g).is_ok()
                && read_generation_file(&recovery_path(dir, g), g).is_ok() {
                let _ = fs::remove_file(generation_path(dir, g));
                let _ = fs::remove_file(recovery_path(dir, g));
            }
        }
    }
    Ok(())
}

impl Gateway {
    pub fn open(data: &Path, cluster_name: &str) -> std::io::Result<Gateway> {
        let dir = data.join("_cluster");
        fs::create_dir_all(&dir)?;
        let legacy = dir.join(LEGACY);
        // A crash between the downgrade fence and generation N's recovery copy
        // leaves only the placeholder: N was never written, so N-1 (or the
        // legacy file / an empty node) is the last acknowledged state.
        let written = written_generations(&dir)?;
        let (persisted, disk) = match written.last() {
            Some(&latest) => {
                let mut recovered = None;
                let mut failures = Vec::new();
                for candidate in written.iter().copied().rev() {
                    match load_generations(&dir, candidate) {
                        Ok(state) => { recovered = Some(state); break; }
                        Err(e) if matches!(e.kind(), std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied) => return Err(e),
                        Err(e) => failures.push(e.to_string()),
                    }
                }
                let (p, mut disk) = recovered.ok_or_else(|| std::io::Error::other(format!(
                    "no intact gateway generation in {}: {}", dir.display(), failures.join("; "))))?;
                if disk.generation != latest {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        GatewayRecoveryRequired { newest_generation: latest,
                            recovered_generation: disk.generation, last_good_state: p, failures }));
                }
                // A crash between the first generation and the marker, or a
                // node from before the marker was always written: an older
                // version must not start from the old file (or none).
                let marked = fs::read(&legacy).ok().is_some_and(|b| serde_json::from_slice::<PersistedState>(&b).is_err());
                if !marked {
                    write_durably(&legacy, &legacy_marker())?;
                }
                disk.legacy_replaced = true;
                (p, disk)
            }
            None => match fs::read(&legacy) {
                Ok(b) => match serde_json::from_slice::<PersistedState>(&b) {
                    Ok(p) => (p, Disk::default()),
                    Err(_) if serde_json::from_slice::<Value>(&b).is_ok_and(|v| v["format"] == "incremental") => {
                        return Err(std::io::Error::other(format!("{} marks an incremental cluster state, but no {STATE_PREFIX}<generation>.json file is left in {}", legacy.display(), dir.display())));
                    }
                    Err(e) => return Err(std::io::Error::other(format!("corrupt {}: {e}", legacy.display()))),
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (PersistedState::empty(cluster_name), Disk::default()),
                Err(e) => return Err(e),
            },
        };
        let mut persisted = persisted;
        // A state changed at load is not the one the master published under
        // this uuid: a fresh uuid makes the next diff to this node fall back
        // to the full state (a diff from the published base would leave
        // the loaded differences in place).
        if settings_as_loaded(&mut persisted.last_accepted_state.metadata)? {
            persisted.last_accepted_state.state_uuid = random_id();
        }
        if persisted.last_accepted_state.cluster_name != cluster_name && persisted.last_accepted_state.version > 0 {
            return Err(std::io::Error::other(format!(
                "this node previously belonged to cluster [{}] and cannot join cluster [{cluster_name}]; remove {} to reset it",
                persisted.last_accepted_state.cluster_name,
                dir.display()
            )));
        }
        Ok(Gateway { dir, persisted, disk })
    }

    pub fn persisted(&self) -> &PersistedState {
        &self.persisted
    }

    /// [`load_committed_segment_format`] for this gateway's data path.
    pub fn committed_segment_format(&self) -> std::io::Result<Option<(String, u32)>> {
        read_committed_segment_format(&self.dir)
    }

    /// Apply and durably write a coordinator `Persist` output.
    pub fn persist(&mut self, update: &PersistUpdate) -> std::io::Result<()> {
        self.persist_all(std::slice::from_ref(update))
    }

    /// Apply several `Persist` outputs in order and write the result once
    /// (the persistence thread's group commit: only the final state needs
    /// to be on disk before any output that follows the last of them).
    /// Only the parts that changed since the last generation are written.
    pub fn persist_all(&mut self, updates: &[PersistUpdate]) -> std::io::Result<()> {
        let Some(last) = updates.last() else { return Ok(()) };
        let term = last.current_term;
        let newest = updates.iter().rev().find_map(|u| u.last_accepted_state.as_deref());
        let old = &self.persisted.last_accepted_state;
        write_generation(&self.dir, &mut self.disk, old, term, newest.unwrap_or(old))?;
        for update in updates {
            update.apply_to(&mut self.persisted);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn im(name: &str) -> IndexMetadata {
        let mut m: IndexMetadata = serde_json::from_value(serde_json::json!({
            "name": name, "uuid": format!("u-{name}"), "version": 1, "mapping_version": 1, "settings_version": 1,
            "aliases_version": 1, "state": "open", "settings": {"index.number_of_shards": "1"}, "mappings": {"properties": {"f": {"type": "keyword"}}}, "aliases": {},
            "number_of_shards": 1, "number_of_replicas": 0, "routing_num_shards": 1,
            "primary_terms": [1], "in_sync_allocation_ids": {}, "creation_date": 0
        }))
        .unwrap_or_else(|e| panic!("index metadata shape: {e}"));
        m.name = name.into();
        m
    }

    fn accepted(term: u64, s: &ClusterState) -> PersistUpdate {
        PersistUpdate { current_term: term, last_accepted_state: Some(Box::new(s.clone())) }
    }

    fn latest(d: &Path) -> Value {
        let dir = d.join("_cluster");
        let g = *generations(&dir).unwrap().last().unwrap();
        let envelope: Value = serde_json::from_slice(&fs::read(generation_path(&dir, g)).unwrap()).unwrap();
        envelope["payload"].clone()
    }

    #[test]
    fn checked_unmirrored_format_upgrades_and_incomplete_upgrade_fails_closed() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut gateway = Gateway::open(d.path(), "c1").unwrap();
        gateway.persist(&PersistUpdate { current_term: 7, last_accepted_state: None }).unwrap();
        let mut payload = latest(d.path());
        payload["format"] = Value::from(CHECKED_FORMAT);
        let bytes = serde_json::to_vec(&payload).unwrap();
        // Preserve exact payload bytes rather than Value's serialization order.
        let checked = CheckedGeneration { format: CHECKED_FORMAT, downgrade_notice: None, crc32: crc32(&bytes),
            payload: serde_json::value::RawValue::from_string(String::from_utf8(bytes).unwrap()).unwrap() };
        fs::write(generation_path(&dir, 1), serde_json::to_vec(&checked).unwrap()).unwrap();
        fs::remove_file(recovery_path(&dir, 1)).unwrap();
        let mut upgraded = Gateway::open(d.path(), "c1").unwrap();
        assert!(upgraded.disk.upgrade);
        upgraded.persist(&PersistUpdate { current_term: 8, last_accepted_state: None }).unwrap();
        assert_eq!(latest(d.path())["format"], FORMAT);
        assert!(recovery_path(&dir, 2).exists());
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 8);
        // Crash at the first write's downgrade fence, before its mirror:
        // generation 3 was never written or acknowledged, so 2 is current.
        fs::write(generation_path(&dir, 3), RECOVERY_PLACEHOLDER).unwrap();
        let mut back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 8);
        back.persist(&PersistUpdate { current_term: 9, last_accepted_state: None }).unwrap();
        assert!(!generation_path(&dir, 3).exists(), "the orphaned placeholder is collected");
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 9);
    }

    /// Crash between the downgrade-fence placeholder and the recovery copy of
    /// a format-1 → 3 upgrade write: generation N-1 is the last acked state.
    #[test]
    fn crash_after_upgrade_placeholder_falls_back_to_previous_generation() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.metadata.indices.insert("a".into(), im("a"));
        state.version = 4;
        g.persist(&accepted(3, &state)).unwrap();
        let mut old = latest(d.path());
        old["format"] = Value::from(LEGACY_FORMAT);
        fs::write(generation_path(&dir, 1), serde_json::to_vec(&old).unwrap()).unwrap();
        fs::remove_file(recovery_path(&dir, 1)).unwrap();
        fs::write(generation_path(&dir, 2), RECOVERY_PLACEHOLDER).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 3);
        assert_eq!(back.persisted().last_accepted_state, state);
        assert!(back.disk.upgrade, "the next write is still the full upgrade write");
        // A placeholder whose recovery copy did become durable is a real
        // generation: the recovery copy wins (crash between the two copies).
        let mut back = back;
        back.persist(&PersistUpdate { current_term: 5, last_accepted_state: None }).unwrap();
        let g = *generations(&dir).unwrap().last().unwrap();
        fs::write(generation_path(&dir, g), RECOVERY_PLACEHOLDER).unwrap();
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 5);
    }

    /// A floor only accepted (in the last accepted state, maybe never
    /// committed) is not the committed floor a restart writes in: that is
    /// only what the applier recorded. The record survives reopening, a
    /// corrupt one fails loudly, and it never shadows a state generation.
    #[test]
    fn the_committed_segment_format_is_not_read_from_the_accepted_state() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.metadata.cluster_uuid = "u".into();
        state.version = 4;
        assert!(elasticsearch_cluster::compatibility::raise_segment_format(&mut state, 13));
        g.persist(&accepted(3, &state)).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(elasticsearch_cluster::compatibility::segment_format_floor(&back.persisted().last_accepted_state.metadata), Some(13));
        assert_eq!(back.committed_segment_format().unwrap(), None, "an accepted raise is not a committed one");
        assert_eq!(load_committed_segment_format(d.path()).unwrap(), None);
        store_committed_segment_format(d.path(), "u", 12).unwrap();
        store_committed_segment_format(d.path(), "u", 13).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().last_accepted_state, state);
        assert_eq!(back.committed_segment_format().unwrap(), Some(("u".to_string(), 13)));
        fs::write(d.path().join("_cluster").join(SEGMENT_FORMAT_FILE), b"{\"floor\":").unwrap();
        assert_eq!(load_committed_segment_format(d.path()).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
    }

    /// A fresh node crashing in its very first write starts empty, rather than
    /// failing with "no intact gateway generation".
    #[test]
    fn crash_after_first_placeholder_on_a_fresh_node_starts_empty() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        fs::create_dir_all(&dir).unwrap();
        fs::write(generation_path(&dir, 1), RECOVERY_PLACEHOLDER).unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(g.persisted().current_term, 0);
        g.persist(&PersistUpdate { current_term: 1, last_accepted_state: None }).unwrap();
        assert!(!generation_path(&dir, 1).exists());
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 1);
    }

    /// Crash in the first write after reading `coordination.json`: the legacy
    /// state is still the last acked one and must not be ignored.
    #[test]
    fn crash_after_placeholder_during_legacy_upgrade_rereads_the_legacy_file() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        fs::create_dir_all(&dir).unwrap();
        let mut old = PersistedState::empty("c1");
        old.current_term = 6;
        old.last_accepted_state.version = 9;
        old.last_accepted_state.metadata.indices.insert("a".into(), im("a"));
        fs::write(dir.join(LEGACY), serde_json::to_vec(&old).unwrap()).unwrap();
        fs::write(generation_path(&dir, 1), RECOVERY_PLACEHOLDER).unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(g.persisted().current_term, 6);
        assert_eq!(g.persisted().last_accepted_state, old.last_accepted_state);
        g.persist(&PersistUpdate { current_term: 7, last_accepted_state: None }).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 7);
        assert_eq!(back.persisted().last_accepted_state, old.last_accepted_state);
    }

    #[test]
    fn damaged_primary_recovers_same_term_and_accepted_value_without_overwriting_evidence() {
        for torn in [false, true] {
            let d = tempfile::tempdir().unwrap();
            let dir = d.path().join("_cluster");
            let mut gateway = Gateway::open(d.path(), "c1").unwrap();
            let mut state = gateway.persisted().last_accepted_state.clone();
            state.version = 1;
            gateway.persist(&accepted(7, &state)).unwrap();
            state.version = 2;
            state.metadata.indices.insert("a".into(), im("a"));
            gateway.persist(&accepted(9, &state)).unwrap();
            let path = generation_path(&dir, 2);
            let mut bytes = fs::read(&path).unwrap();
            if torn { bytes.truncate(bytes.len() / 2); }
            else {
                let needle = b"\"current_term\":9";
                let at = bytes.windows(needle.len()).position(|w| w == needle).unwrap() + needle.len() - 1;
                bytes[at] = b'8';
                assert!(serde_json::from_slice::<Value>(&bytes).is_ok());
            }
            fs::write(&path, &bytes).unwrap();
            let mut recovered = Gateway::open(d.path(), "c1").unwrap();
            assert_eq!(recovered.persisted().current_term, 9);
            assert_eq!(recovered.persisted().last_accepted_state, state);
            assert_eq!(fs::read(&path).unwrap(), bytes);
            recovered.persist(&PersistUpdate { current_term: 10, last_accepted_state: None }).unwrap();
            assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 10);
            assert_eq!(fs::read(&path).unwrap(), bytes, "preserve rejected primary evidence");
        }
    }

    #[test]
    fn mirror_only_crash_and_corrupt_referenced_entry_recover_latest_state() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut gateway = Gateway::open(d.path(), "c1").unwrap();
        let mut state = gateway.persisted().last_accepted_state.clone();
        state.version = 1;
        state.metadata.indices.insert("a".into(), im("a"));
        gateway.persist(&accepted(4, &state)).unwrap();
        gateway.persist(&PersistUpdate { current_term: 5, last_accepted_state: None }).unwrap();
        fs::remove_file(generation_path(&dir, 2)).unwrap();
        fs::write(generation_path(&dir, 1), b"torn").unwrap();
        let recovered = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(recovered.persisted().current_term, 5);
        assert_eq!(recovered.persisted().last_accepted_state, state);
        assert!(!generation_path(&dir, 2).exists());
    }

    #[test]
    fn corrupt_recovery_copy_cannot_hide_valid_primary_or_authorize_older_fallback() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut gateway = Gateway::open(d.path(), "c1").unwrap();
        gateway.persist(&PersistUpdate { current_term: 4, last_accepted_state: None }).unwrap();
        gateway.persist(&PersistUpdate { current_term: 5, last_accepted_state: None }).unwrap();
        fs::write(recovery_path(&dir, 2), b"torn").unwrap();
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 5);
        fs::write(generation_path(&dir, 2), b"torn").unwrap();
        let error = Gateway::open(d.path(), "c1").err().unwrap();
        assert!(error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().is_some());
    }

    #[test]
    fn torn_generation_exposes_last_good_but_fences_coordinator_startup() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.version = 1;
        state.metadata.indices.insert("a".into(), im("a"));
        g.persist(&accepted(1, &state)).unwrap();
        state.version = 2;
        state.metadata.indices.get_mut("a").unwrap().version = 2;
        g.persist(&accepted(2, &state)).unwrap();
        let path = generation_path(&dir, 2);
        let bytes = fs::read(&path).unwrap();
        let torn = &bytes[..bytes.len() / 2];
        fs::write(&path, torn).unwrap();
        fs::remove_file(recovery_path(&dir, 2)).unwrap();
        let error = Gateway::open(d.path(), "c1").err().expect("unsafe startup must be fenced");
        let recovery = error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().unwrap();
        assert_eq!(recovery.newest_generation, 2);
        assert_eq!(recovery.recovered_generation, 1);
        assert_eq!(recovery.last_good_state().current_term, 1);
        assert_eq!(recovery.last_good_state().last_accepted_state.version, 1);
        assert_eq!(recovery.last_good_state().last_accepted_state.metadata.indices["a"].version, 1);
        assert_eq!(fs::read(path).unwrap(), torn, "do not overwrite the corrupt generation");
        assert!(!generation_path(&dir, 3).exists(), "recovery inspection is read-only");
        assert!(error.to_string().contains("coordinator startup is fenced"));
    }

    #[test]
    fn same_term_accepted_state_loss_is_also_fenced() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.version = 1;
        g.persist(&accepted(7, &state)).unwrap();
        state.version = 2;
        g.persist(&accepted(7, &state)).unwrap();
        fs::write(generation_path(&dir, 2), b"{}").unwrap();
        fs::remove_file(recovery_path(&dir, 2)).unwrap();
        let error = Gateway::open(d.path(), "c1").err().expect("accepted values cannot roll back even in the same term");
        let recovery = error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().unwrap();
        assert_eq!(recovery.last_good_state().current_term, 7);
        assert_eq!(recovery.last_good_state().last_accepted_state.version, 1);
    }

    #[test]
    fn valid_json_bit_flip_in_referenced_generation_is_detected() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.version = 1;
        state.metadata.indices.insert("a".into(), im("a"));
        g.persist(&accepted(1, &state)).unwrap();
        state.version = 2;
        state.metadata.indices.get_mut("a").unwrap().version = 2;
        g.persist(&accepted(2, &state)).unwrap();
        g.persist(&PersistUpdate { current_term: 3, last_accepted_state: None }).unwrap();
        // The newest root's checksum is intact, but its index entry lives in 2.
        let path = generation_path(&dir, 2);
        let mut bytes = fs::read(&path).unwrap();
        let needle = b"\"current_term\":2";
        let at = bytes.windows(needle.len()).position(|w| w == needle).unwrap() + needle.len() - 1;
        bytes[at] ^= 1; // ASCII 2 -> 3: still valid JSON and a plausible value.
        assert!(serde_json::from_slice::<Value>(&bytes).is_ok());
        fs::write(path, &bytes).unwrap();
        fs::remove_file(recovery_path(&dir, 2)).unwrap();
        let error = read_generation(&dir, 2).err().unwrap().to_string();
        assert!(error.contains("checksum"), "{error}");
        let error = Gateway::open(d.path(), "c1").err().expect("corrupt dependency fences startup");
        let recovery = error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().unwrap();
        assert_eq!(recovery.last_good_state().current_term, 1);
        assert_eq!(recovery.last_good_state().last_accepted_state.metadata.indices["a"].version, 1);
    }

    #[test]
    fn compaction_retains_previous_generation_dependencies_for_fallback() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        for i in 0..MAX_REFERENCED + 2 {
            let name = format!("i{i}");
            state.metadata.indices.insert(name.clone(), im(&name));
        }
        g.persist(&accepted(1, &state)).unwrap();
        for i in 0..MAX_REFERENCED {
            state.version += 1;
            state.metadata.indices.get_mut(&format!("i{i}")).unwrap().version += 1;
            g.persist(&accepted(1, &state)).unwrap();
        }
        let previous = state.clone();
        let previous_generation = g.disk.generation;
        state.version += 1;
        g.persist(&accepted(2, &state)).unwrap();
        assert!(latest(d.path())["indices"].as_object().unwrap().values().all(|e| e.get("inline").is_some()));
        fs::write(generation_path(&dir, g.disk.generation), b"{\"format\":2,").unwrap();
        fs::remove_file(recovery_path(&dir, g.disk.generation)).unwrap();
        let error = Gateway::open(d.path(), "c1").err().expect("compaction corruption fences startup");
        let recovery = error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().unwrap();
        assert_eq!(recovery.recovered_generation, previous_generation);
        assert_eq!(recovery.last_good_state().last_accepted_state, previous);
    }

    #[test]
    fn unfinished_temp_is_ignored_but_no_intact_generation_fails_closed() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        g.persist(&PersistUpdate { current_term: 4, last_accepted_state: None }).unwrap();
        fs::write(dir.join("state-2.tmp"), b"{\"format\":2,").unwrap();
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 4);
        fs::write(generation_path(&dir, 1), b"{}").unwrap();
        fs::remove_file(recovery_path(&dir, 1)).unwrap();
        let error = Gateway::open(d.path(), "c1").err().unwrap().to_string();
        assert!(error.contains("no intact gateway generation"), "{error}");
    }

    #[test]
    fn unknown_newer_format_refuses_downgrade_instead_of_falling_back() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        g.persist(&PersistUpdate { current_term: 1, last_accepted_state: None }).unwrap();
        g.persist(&PersistUpdate { current_term: 2, last_accepted_state: None }).unwrap();
        // A newer writer: checksummed payload and envelope agree on format 999.
        let mut payload = latest(d.path());
        payload["format"] = Value::from(999);
        let bytes = serde_json::to_vec(&payload).unwrap();
        let newer = serde_json::to_vec(&serde_json::json!({"format": 999, "crc32": crc32(&bytes),
            "payload": serde_json::value::RawValue::from_string(String::from_utf8(bytes).unwrap()).unwrap()})).unwrap();
        for path in [generation_path(&dir, 2), recovery_path(&dir, 2)] { fs::write(path, &newer).unwrap(); }
        let error = Gateway::open(d.path(), "c1").err().unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::Unsupported, "{error}");
        assert!(error.to_string().contains("refusing a downgrade"), "{error}");
    }

    /// The envelope's format is outside the CRC: one flipped bit must not
    /// brick the node with a downgrade message while an intact copy exists.
    #[test]
    fn flipped_envelope_format_uses_the_recovery_copy_not_a_downgrade_refusal() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        g.persist(&PersistUpdate { current_term: 1, last_accepted_state: None }).unwrap();
        g.persist(&PersistUpdate { current_term: 2, last_accepted_state: None }).unwrap();
        let primary = fs::read(generation_path(&dir, 2)).unwrap();
        let flipped = String::from_utf8(primary.clone()).unwrap().replacen("\"format\":3", "\"format\":7", 1);
        assert_ne!(flipped.as_bytes(), &primary[..]);
        fs::write(generation_path(&dir, 2), &flipped).unwrap();
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 2);
        assert_eq!(fs::read_to_string(generation_path(&dir, 2)).unwrap(), flipped, "evidence is not overwritten");
        // A bare unknown header is also unverifiable corruption.
        fs::write(generation_path(&dir, 2), b"{\"format\":999}").unwrap();
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 2);
        // With both copies damaged, the older generation is offered for
        // recovery but startup stays fenced: it is not reported as a downgrade.
        fs::write(recovery_path(&dir, 2), &flipped).unwrap();
        let error = Gateway::open(d.path(), "c1").err().unwrap();
        assert_ne!(error.kind(), std::io::ErrorKind::Unsupported, "{error}");
        let recovery = error.get_ref().unwrap().downcast_ref::<GatewayRecoveryRequired>().unwrap();
        assert_eq!(recovery.last_good_state().current_term, 1);
    }

    /// The master baseline's reader (format 1) must fail with an explicit
    /// downgrade refusal on every file a checked writer leaves in `state-N`.
    #[test]
    fn baseline_reader_reports_an_explicit_downgrade_refusal() {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct BaselineGenerationIn { format: u32, generation: u64, current_term: u64, state: ClusterState,
            global: Entry<Globals>, indices: BTreeMap<String, Entry<IndexEntry>> }
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        g.persist(&PersistUpdate { current_term: 1, last_accepted_state: None }).unwrap();
        for bytes in [fs::read(generation_path(&dir, 1)).unwrap(), RECOVERY_PLACEHOLDER.to_vec()] {
            let error = serde_json::from_slice::<BaselineGenerationIn>(&bytes).err().unwrap().to_string();
            assert!(error.contains("downgrading this node is not supported"), "{error}");
        }
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 1);
    }

    #[test]
    fn unchecked_incremental_format_is_read_and_fully_upgraded() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        state.metadata.indices.insert("a".into(), im("a"));
        g.persist(&accepted(1, &state)).unwrap();
        let mut old = latest(d.path());
        old["format"] = Value::from(LEGACY_FORMAT);
        fs::write(generation_path(&dir, 1), serde_json::to_vec(&old).unwrap()).unwrap();
        let mut back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().last_accepted_state, state);
        back.persist(&PersistUpdate { current_term: 2, last_accepted_state: None }).unwrap();
        let upgraded = latest(d.path());
        assert_eq!(upgraded["format"], FORMAT);
        assert!(upgraded["global"].get("inline").is_some());
        assert!(upgraded["indices"]["a"].get("inline").is_some());
        // Legacy parser refuses format 2 rather than loading stale state.
        let envelope: Value = serde_json::from_slice(&fs::read(generation_path(&dir, 2)).unwrap()).unwrap();
        assert_eq!(envelope["format"], FORMAT);
        assert_eq!(Gateway::open(d.path(), "c1").unwrap().persisted().current_term, 2);
    }

    #[cfg(unix)]
    #[test]
    fn the_node_lock_is_exclusive_across_processes() {
        let d = tempfile::tempdir().unwrap();
        lock_data_path(d.path(), std::time::Duration::ZERO).unwrap();
        // Again from this process: already held, not locked twice.
        lock_data_path(d.path(), std::time::Duration::ZERO).unwrap();
        // Another open file description (as another process holds): refused.
        let other = fs::OpenOptions::new().write(true).open(d.path().join("node.lock")).unwrap();
        assert!(!try_lock_exclusive(&other).unwrap(), "a second holder got the node lock");
    }

    #[test]
    fn node_id_is_stable() {
        let d = tempfile::tempdir().unwrap();
        let a = load_or_create_node_id(d.path()).unwrap();
        let b = load_or_create_node_id(d.path()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 22);
    }

    #[test]
    fn persist_roundtrip_and_cluster_name_guard() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut s = g.persisted().last_accepted_state.clone();
        s.version = 7;
        g.persist(&PersistUpdate { current_term: 3, last_accepted_state: Some(Box::new(s)) }).unwrap();
        let g2 = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(g2.persisted().current_term, 3);
        assert_eq!(g2.persisted().last_accepted_state.version, 7);
        assert!(Gateway::open(d.path(), "other").is_err());
    }

    #[test]
    fn grouped_updates_persist_the_final_state() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut s = g.persisted().last_accepted_state.clone();
        s.version = 4;
        let accepted = PersistUpdate { current_term: 2, last_accepted_state: Some(Box::new(s)) };
        let term_only = PersistUpdate { current_term: 5, last_accepted_state: None };
        g.persist_all(&[accepted, term_only]).unwrap();
        let g2 = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(g2.persisted().current_term, 5);
        assert_eq!(g2.persisted().last_accepted_state.version, 4, "a term-only update keeps the accepted state");
    }

    /// Review M11: a publication that changes one index writes that index
    /// (and the small rest of the state), not every index; the state reads
    /// back exactly, and unreferenced generations are deleted.
    #[test]
    fn only_changed_indices_are_rewritten() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut s = g.persisted().last_accepted_state.clone();
        s.version = 1;
        for n in ["a", "b", "c"] {
            s.metadata.indices.insert(n.into(), im(n));
            s.routing_table.indices.insert(n.into(), IndexRoutingTable { index_uuid: format!("u-{n}"), shards: vec![] });
        }
        s.metadata.persistent_settings.insert("cluster.routing.allocation.enable".into(), Value::String("all".into()));
        s.metadata.index_templates.insert("t".into(), serde_json::json!({"index_patterns": ["x*"]}));
        g.persist(&accepted(1, &s)).unwrap();
        let first = latest(d.path());
        assert!(first["indices"].as_object().unwrap().values().all(|e| e.get("inline").is_some()), "the first generation is full");
        // Only `b` changes.
        s.version = 2;
        s.metadata.indices.get_mut("b").unwrap().version = 2;
        g.persist(&accepted(1, &s)).unwrap();
        let second = latest(d.path());
        assert_eq!(second["generation"], 2);
        assert!(second["indices"]["b"].get("inline").is_some());
        assert_eq!(second["indices"]["a"], serde_json::json!({"in": 1}));
        assert_eq!(second["indices"]["c"], serde_json::json!({"in": 1}));
        assert_eq!(second["global"], serde_json::json!({"in": 1}), "unchanged global metadata is not rewritten");
        // A term-only update and a new template.
        g.persist(&PersistUpdate { current_term: 4, last_accepted_state: None }).unwrap();
        s.metadata.index_templates.insert("t2".into(), serde_json::json!({"index_patterns": ["y*"]}));
        s.metadata.indices.remove("c");
        s.routing_table.indices.remove("c");
        s.version = 3;
        g.persist(&accepted(4, &s)).unwrap();
        let fourth = latest(d.path());
        assert!(fourth["global"].get("inline").is_some());
        assert!(fourth["indices"].get("c").is_none());
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 4);
        assert_eq!(back.persisted().last_accepted_state, s, "the state reads back exactly");
        // Generation 1 still holds `a`; 3 (term only) is referenced by nothing.
        let gens = generations(&d.path().join("_cluster")).unwrap();
        assert_eq!(gens, vec![1, 2, 3, 4], "the previous root and its dependencies are retained");
    }

    /// Combined review r2 (M30): a restart from disk drops transient
    /// settings (ES never persists them) and archives persistent ones this
    /// version does not know or whose value no longer validates.
    #[test]
    fn a_restart_drops_transient_and_archives_unknown_settings() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut s = g.persisted().last_accepted_state.clone();
        s.version = 1;
        s.metadata.persistent_settings.insert("cluster.routing.allocation.enable".into(), Value::String("all".into()));
        s.metadata.persistent_settings.insert("cluster.gone_in_this_version".into(), Value::String("x".into()));
        s.metadata.persistent_settings.insert("search.max_keep_alive".into(), Value::String("abc".into()));
        s.metadata.transient_settings.insert("indices.breaker.request.limit".into(), Value::String("1mb".into()));
        g.persist(&accepted(1, &s)).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        let md = &back.persisted().last_accepted_state.metadata;
        assert!(md.transient_settings.is_empty(), "{:?}", md.transient_settings);
        assert_eq!(md.persistent_settings["cluster.routing.allocation.enable"], "all");
        assert_eq!(md.persistent_settings["archived.cluster.gone_in_this_version"], "x");
        assert_eq!(md.persistent_settings["archived.search.max_keep_alive"], "abc");
        assert!(md.persistent_settings.get("search.max_keep_alive").is_none());
    }

    #[test]
    fn restart_archives_invalid_index_values_without_rejecting_legacy_scalars() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut state = g.persisted().last_accepted_state.clone();
        let mut index = im("legacy");
        index.settings.insert("index.max_regex_length".into(), serde_json::json!("broken"));
        index.settings.insert("index.refresh_interval".into(), serde_json::json!("bad-unit"));
        index.settings.insert("index.number_of_replicas".into(), serde_json::json!(0));
        index.settings.insert("index.analysis.analyzer.custom.tokenizer".into(), serde_json::json!("standard"));
        index.settings.insert("index.query.default_field.0".into(), serde_json::json!("first"));
        index.settings.insert("index.query.default_field.1".into(), serde_json::json!("second"));
        state.metadata.indices.insert(index.name.clone(), index);
        state.version = 1;
        g.persist(&accepted(1, &state)).unwrap();
        drop(g);
        let mut reopened = Gateway::open(d.path(), "c1").unwrap();
        let upgraded = reopened.persisted().last_accepted_state.clone();
        let settings = &upgraded.metadata.indices["legacy"].settings;
        assert_eq!(settings["archived.index.max_regex_length"], "broken");
        assert_eq!(settings["archived.index.refresh_interval"], "bad-unit");
        assert!(!settings.contains_key("index.max_regex_length"));
        assert!(!settings.contains_key("index.refresh_interval"));
        assert_eq!(settings["index.number_of_replicas"], 0);
        assert_eq!(settings["index.analysis.analyzer.custom.tokenizer"], "standard");
        assert_eq!(settings["index.query.default_field"], serde_json::json!(["first", "second"]));
        assert!(!settings.contains_key("index.query.default_field.0"));
        assert_ne!(upgraded.state_uuid, state.state_uuid, "old publication diffs cannot reuse the upgraded base");
        reopened.persist(&accepted(2, &upgraded)).unwrap();
        drop(reopened);
        let again = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(again.persisted().last_accepted_state.metadata.indices["legacy"].settings, upgraded.metadata.indices["legacy"].settings);
    }

    #[test]
    fn many_referenced_generations_are_compacted() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        let mut s = g.persisted().last_accepted_state.clone();
        let names: Vec<String> = (0..(MAX_REFERENCED + 4)).map(|i| format!("i{i}")).collect();
        for n in &names {
            s.metadata.indices.insert(n.clone(), im(n));
        }
        g.persist(&accepted(1, &s)).unwrap();
        // Each generation changes another index: references pile up until
        // a full generation is written.
        for (v, n) in names.iter().enumerate() {
            s.version = v as u64 + 2;
            s.metadata.indices.get_mut(n).unwrap().version += 1;
            g.persist(&accepted(1, &s)).unwrap();
            assert!(generations(&d.path().join("_cluster")).unwrap().len() <= 2 * (MAX_REFERENCED + 2));
        }
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().last_accepted_state, s);
    }

    /// Review round 3 (NEW-1, NEW-2): a node that never had the old file gets
    /// the marker with its first generation, and a node whose old file
    /// survived next to a generation (a crash in between) gets it at open.
    #[test]
    fn the_marker_is_always_written() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        g.persist(&PersistUpdate { current_term: 2, last_accepted_state: None }).unwrap();
        assert!(serde_json::from_slice::<PersistedState>(&fs::read(dir.join(LEGACY)).unwrap()).is_err(), "a fresh node writes the marker");
        // A stale old file next to a generation (crash before the marker).
        let mut stale = PersistedState::empty("c1");
        stale.current_term = 1;
        fs::write(dir.join(LEGACY), serde_json::to_vec(&stale).unwrap()).unwrap();
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 2, "the generation wins");
        assert!(serde_json::from_slice::<PersistedState>(&fs::read(dir.join(LEGACY)).unwrap()).is_err(), "the stale file is replaced");
        // Only the marker left: a clear refusal.
        for g in generations(&dir).unwrap() {
            fs::remove_file(generation_path(&dir, g)).unwrap();
            let _ = fs::remove_file(recovery_path(&dir, g));
        }
        let err = Gateway::open(d.path(), "c1").err().expect("refused").to_string();
        assert!(err.contains("incremental"), "{err}");
    }

    /// Upgrade: the previous single-file format is read once, then replaced
    /// by a marker an older version refuses to start on.
    #[test]
    fn the_legacy_file_is_read_then_replaced_by_a_marker() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("_cluster");
        fs::create_dir_all(&dir).unwrap();
        let mut old = PersistedState::empty("c1");
        old.current_term = 6;
        old.last_accepted_state.version = 9;
        old.last_accepted_state.metadata.indices.insert("a".into(), im("a"));
        fs::write(dir.join(LEGACY), serde_json::to_vec(&old).unwrap()).unwrap();
        let mut g = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(g.persisted().current_term, 6);
        assert_eq!(g.persisted().last_accepted_state, old.last_accepted_state);
        g.persist(&PersistUpdate { current_term: 7, last_accepted_state: None }).unwrap();
        assert!(serde_json::from_slice::<PersistedState>(&fs::read(dir.join(LEGACY)).unwrap()).is_err(), "an older version cannot read the marker");
        let back = Gateway::open(d.path(), "c1").unwrap();
        assert_eq!(back.persisted().current_term, 7);
        assert_eq!(back.persisted().last_accepted_state, old.last_accepted_state);
    }
}
