//! Dynamic mapping updates, as ES's `TransportShardBulkAction` +
//! `MappingUpdatedAction` handle them: a new field is committed by the
//! master *before* any copy indexes a document that needs it.
//!
//! **Primary side** ([`ClusterMappingHook`], the engine's A6 hook installed
//! with `Node::set_mapping_update_hook` on every engine of the node). The
//! engine defers every document of a shard request whose fields are not all
//! mapped, proposes the combined mapping of all of them (in request order,
//! whether they were parsed sequentially or in parallel) and calls the hook
//! once per shard request (`IndexShard::write_inner`). The hook does not
//! apply anything. The hook sends it to the master and blocks the indexing
//! thread (which holds the shard's operation permit, as ES's bulk shard
//! request does) until the committed cluster state carrying it has been
//! applied on this node, bounded by `indices.mapping.dynamic_timeout`
//! (30s). Then the engine re-parses the document against the applied
//! mapping: the field is mapped with the master's type, or the document
//! fails with ES's `document_parsing_exception` if a concurrent update
//! mapped it with another type first. On timeout or no master the hook
//! answers 503 (`process_cluster_event_timeout_exception`) and only the
//! documents that needed the update fail; on a rejection each of them
//! retries on its own (per-document proposal, ES's error for a loser); everything already
//! mapped indexes without ever contacting the master. Identical concurrent
//! proposals from any shard of the node share one request.
//!
//! **Replica side.** Replication requests carry the primary's mapping
//! ahead of the operations (`ReplicationRequest::mapping_json`), and the
//! primary's mapping only ever holds committed fields, so a replica has
//! the mapping before it applies an operation that uses it.
//!
//! **Master side** ([`handle`]). Updates are queued to the metadata
//! service (`master_service`): every update queued while a publication is
//! in flight goes into the next batch, identical ones collapse, conflicting
//! ones resolve in queue order (first wins; the later one gets the merge
//! error). The reply is the committed version; as in ES (`ackTimeout` 0)
//! the master does not wait for other nodes to apply it, the requesting
//! primary waits for its own node.
//!
//! Two races of the older handler stay handled: the master's own engine
//! copy of the index can be briefly missing while it rebuilds a replica
//! (`wipe_local_copy`) — the handler waits for it; and a re-put the engine
//! rejects (e.g. dense_vector) is accepted when the published mapping
//! already is the requested one.

use crate::node::{ClusterNode, MappingUpdateReq};
use elasticsearch_engine::error::EsError;
use elasticsearch_transport::RemoteError;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

/// Separate action/schema: never encode a safety-critical UUID as an optional
/// trailing field that an older receiver is allowed to ignore.
#[derive(serde::Serialize)]
pub(crate) struct UuidMappingUpdate {
    index: String,
    mapping_json: String,
    index_uuid: String,
    dynamic_delta: Option<String>,
    #[serde(skip)]
    memory: crate::replication::decode::DecodeMemory,
}
impl UuidMappingUpdate {
    fn from_request(req: &MappingUpdateReq) -> Result<Self, RemoteError> {
        let index_uuid = require_uuid(req.index_uuid.as_deref())?;
        let memory=crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[&req.index,&req.mapping_json,index_uuid,req.dynamic_delta.as_deref().unwrap_or("")]).map_err(|e|remote(&e.kind,e.reason,e.status))?;
        Ok(Self { index: copy_wire_text(&req.index).map_err(|e|remote(&e.kind,e.reason,e.status))?, mapping_json: copy_wire_text(&req.mapping_json).map_err(|e|remote(&e.kind,e.reason,e.status))?, index_uuid: copy_wire_text(index_uuid).map_err(|e|remote(&e.kind,e.reason,e.status))?, dynamic_delta:req.dynamic_delta.as_deref().map(copy_wire_text).transpose().map_err(|e|remote(&e.kind,e.reason,e.status))?, memory })
    }
    pub(crate) fn into_request(self) -> Result<MappingUpdateReq, RemoteError> {
        require_uuid(Some(&self.index_uuid))?;
        Ok(MappingUpdateReq { index: self.index, mapping_json: self.mapping_json, index_uuid: Some(self.index_uuid), dynamic_delta:self.dynamic_delta,memory: self.memory })
    }
}

impl<'de> serde::Deserialize<'de> for UuidMappingUpdate {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        use serde::de::{Error,SeqAccess,Visitor};
        use crate::replication::decode::{Text,OptionalText,DecodeMemory};
        if d.is_human_readable() {
            let request=<MappingUpdateReq as serde::Deserialize>::deserialize(d)?;
            let index_uuid=request.index_uuid.ok_or_else(||D::Error::missing_field("index_uuid"))?;
            return Ok(Self{index:request.index,mapping_json:request.mapping_json,index_uuid,dynamic_delta:request.dynamic_delta,memory:request.memory});
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=UuidMappingUpdate;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.write_str("UUID-fenced mapping update")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error> {
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"UUID mapping envelope").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::invalid_length(0,&self))?;
                let mapping_json=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::invalid_length(1,&self))?;
                let index_uuid=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::invalid_length(2,&self))?;
                let dynamic_delta=seq.next_element_seed(OptionalText(&mut memory))?.ok_or_else(||A::Error::invalid_length(3,&self))?;
                Ok(UuidMappingUpdate{index,mapping_json,index_uuid,dynamic_delta,memory:DecodeMemory::new(memory)})
            }
        }
        d.deserialize_struct("UuidMappingUpdate", &["index","mapping_json","index_uuid","dynamic_delta"], V)
    }
}
pub(crate) fn copy_wire_text(text:&str)->elasticsearch_engine::error::Result<String> {
    let mut out=String::new();
    out.try_reserve_exact(text.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",format!("mapping text allocation: {e}")))?;
    out.push_str(text);Ok(out)
}
fn encode_mapping_request(index:&str,mapping:&Value,uuid:Option<&str>,dynamic_delta:Option<&str>)->elasticsearch_engine::error::Result<MappingUpdateReq> {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{self.0=self.0.checked_add(bytes.len()).ok_or_else(||std::io::Error::other("mapping length overflow"))?;Ok(bytes.len())}
        fn flush(&mut self)->std::io::Result<()>{Ok(())}
    }
    let mut count=Count(0);
    serde_json::to_writer(&mut count,mapping).map_err(|e|EsError::new(500,"serialization_exception",e.to_string()))?;
    let bytes=index.len().checked_add(uuid.map_or(0,str::len)).and_then(|n|n.checked_add(count.0)).and_then(|n|n.checked_add(dynamic_delta.map_or(0,str::len))).ok_or_else(||EsError::new(429,"circuit_breaking_exception","mapping size overflow"))?;
    let memory=elasticsearch_engine::memory::breakers().reserve_request(bytes as u64,"mapping envelope producer")?;
    let mut encoded=Vec::new();encoded.try_reserve_exact(count.0).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;encoded.resize(count.0,0);
    let mut writer=std::io::Cursor::new(encoded.as_mut_slice());
    serde_json::to_writer(&mut writer,mapping).map_err(|e|EsError::new(500,"serialization_exception",e.to_string()))?;
    if writer.position()!=count.0 as u64 {return Err(EsError::new(500,"serialization_exception","mapping size changed"));}
    Ok(MappingUpdateReq{index:copy_wire_text(index)?,mapping_json:String::from_utf8(encoded).map_err(|e|EsError::new(500,"serialization_exception",e.to_string()))?,index_uuid:uuid.map(copy_wire_text).transpose()?,dynamic_delta:dynamic_delta.map(copy_wire_text).transpose()?,memory:crate::replication::decode::DecodeMemory::new(memory)})
}

fn require_uuid(uuid: Option<&str>) -> Result<&str, RemoteError> {
    uuid.filter(|s| !s.is_empty()).ok_or_else(|| remote("unsupported_operation_exception",
        "dynamic mapping update is missing its index UUID fence; upgrade the sending node", 503))
}

/// How long the master waits for its engine copy of the index to reappear.
const INDEX_WAIT: Duration = Duration::from_secs(10);
/// `indices.mapping.dynamic_timeout` default.
pub const DEFAULT_DYNAMIC_TIMEOUT: Duration = Duration::from_secs(30);

fn remote(kind: &str, reason: impl Into<String>, status: u16) -> RemoteError {
    RemoteError::new(kind, reason.into(), status)
}

/// Handle `ACTION_MAPPING_UPDATE` on the master; resolves with the
/// committed cluster-state version that holds the mapping.
pub async fn handle(n: &Arc<ClusterNode>, r: MappingUpdateReq) -> Result<u64, RemoteError> {
    if !n.is_master() {
        return Err(remote("not_master_exception", "node is not the elected master", 503));
    }
    // Legacy senders without the incarnation fence must not mutate a recreated
    // index. Already-mapped writes do not use this action and remain available.
    require_uuid(r.index_uuid.as_deref())?;
    let want = crate::owned_json::OwnedJson::parse(r.dynamic_delta.as_deref().unwrap_or(&r.mapping_json)).map_err(|e| remote(&e.kind, e.reason, e.status))?;
    let deadline = Instant::now() + INDEX_WAIT;
    loop {
        if n.engine.get_index(&r.index).is_some() {
            break;
        }
        if !n.state().metadata.indices.contains_key(&r.index) || Instant::now() >= deadline {
            return Err(remote("index_not_found_exception", format!("no such index [{}]", r.index), 404));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    // Merged into the master's engine and published in a metadata-service
    // batch, together with every other update queued meanwhile.
    let want = want.attach_wire_owner(r.memory);
    match r.dynamic_delta {
        Some(delta) => {
            drop(delta);
            n.metadata_service().put_mapping_delta(r.index, want, r.index_uuid).await
        }
        None => n.metadata_service().put_mapping(r.index, want, r.index_uuid).await,
    }
}

/// `indices.mapping.dynamic_timeout` from the cluster settings.
pub fn dynamic_timeout(state: &elasticsearch_cluster::state::ClusterState) -> Duration {
    state
        .metadata
        .setting("indices.mapping.dynamic_timeout")
        .and_then(Value::as_str)
        .and_then(|v| elasticsearch_cluster::coordination::parse_time_millis(v).ok())
        .map_or(DEFAULT_DYNAMIC_TIMEOUT, Duration::from_millis)
}

/// The error of a document whose mapping update did not commit in time:
/// ES's `ProcessClusterEventTimeoutException` (503) for the put-mapping
/// task. The engine's batched path (`IndexShard::write_inner`) fails only
/// the items that needed the update on a 503 / timeout from the hook; any
/// other error makes each item retry on its own.
fn timed_out(index: &str, timeout: Duration) -> EsError {
    EsError::new(503, "process_cluster_event_timeout_exception", format!("failed to process cluster event (put-mapping [{index}]) within {}ms (timed out while waiting for a dynamic mapping update)", timeout.as_millis()))
}

/// Errors that mean "the master could not be reached / did not answer",
/// reported as the dynamic-mapping timeout; anything else (a rejection by
/// the master, a failed commit) is passed on as the master gave it.
fn master_unavailable(e: &RemoteError) -> bool {
    matches!(e.error_type.as_str(), "master_not_discovered_exception" | "not_master_exception" | "node_closed_exception")
        || e.error_type.contains("connect") || e.error_type.contains("timeout") || e.error_type.contains("disconnected")
        || e.reason.contains("timed out")
}

type Waiters = Vec<std::sync::mpsc::Sender<Result<(), EsError>>>;

/// The node's dynamic-mapping hook (see the module docs).
pub struct ClusterMappingHook {
    node: Weak<ClusterNode>,
    runtime: tokio::runtime::Handle,
    /// Proposals in flight, by index + proposed mapping.
    inflight: Arc<Mutex<HashMap<String, Waiters>>>,
    /// Distinct proposals being sent to the master (joiners of one in
    /// flight do not count); more queue for a slot (bounded by the dynamic
    /// timeout), as ES queues put-mapping tasks. Taken by the async commit,
    /// never by a write thread.
    proposing: Arc<tokio::sync::Semaphore>,
}

impl ClusterMappingHook {
    /// Must be called on the node's runtime.
    pub fn new(node: Weak<ClusterNode>) -> ClusterMappingHook {
        ClusterMappingHook { node, runtime: tokio::runtime::Handle::current(), inflight: Default::default(), proposing: Arc::new(tokio::sync::Semaphore::new(MAX_PROPOSALS)) }
    }
}

/// Distinct proposals in flight per node. Only a guard against unbounded
/// requests to the master: the master batches every queued proposal into
/// one publication, so a small cap would cap the batch (a wave of new
/// fields would take one publication per few fields). Threads waiting on
/// a proposal are bounded by the write pools, not by this.
const MAX_PROPOSALS: usize = 256;

/// Only the mappers `proposed` adds to `local` (ES's `MappingUpdatedAction`
/// sends the dynamic update, not the whole mapping): new fields (at any
/// depth, an existing object keeping its type) and new runtime fields.
/// Root parameters, `_meta`, dynamic templates and existing fields'
/// mutable parameters never travel; learned dimensions/model settings do. A proposal built on a stale local mapping
/// cannot revert a newer committed change (`dynamic: strict`, a raised
/// `ignore_above`, ...). `None`: nothing new.
pub fn added_mappers(local: &Value, proposed: &Value) -> Option<Value> {
    let mut out = serde_json::Map::new();
    if let Some(props) = added_properties(&local["properties"], &proposed["properties"]) {
        out.insert("properties".into(), props);
        if let Some(value) = local.get("subobjects") { out.insert("subobjects".into(), value.clone()); }
    }
    if let Some(rt) = proposed.get("runtime").and_then(Value::as_object) {
        let have = local.get("runtime").and_then(Value::as_object);
        let add: serde_json::Map<String, Value> = rt.iter().filter(|(k, _)| have.is_none_or(|h| !h.contains_key(*k))).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !add.is_empty() {
            out.insert("runtime".into(), Value::Object(add));
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// The master's view of a dynamic proposal against the `committed`
/// mapping: like [`added_mappers`], but an existing field whose type the
/// proposal changes keeps that type, so the merge rejects the conflict
/// (the first of two concurrent proposals wins). A proposal from an older
/// primary (which sends its whole local mapping) or built on a stale one
/// can so add fields but never revert root or field parameters.
pub fn additive_update(committed: &Value, proposed: &Value) -> Value {
    fn props(have: &Value, proposed: &Value) -> Option<Value> {
        let p = proposed.as_object()?;
        let ty = |d: &Value| d.get("type").and_then(Value::as_str).unwrap_or("object").to_string();
        let mut out = serde_json::Map::new();
        for (name, def) in p {
            let Some(h) = have.get(name) else {
                out.insert(name.clone(), def.clone());
                continue;
            };
            if ty(h) != ty(def) {
                out.insert(name.clone(), serde_json::json!({"type": ty(def)}));
            } else {
                let mut d = learned_field(h, def).and_then(|v| v.as_object().cloned()).unwrap_or_default();
                for key in ["properties", "fields"] {
                    if let Some(sub) = props(&h[key], &def[key]) { d.insert(key.into(), sub); }
                }
                if !d.is_empty() {
                    copy_object_context(h, &mut d);
                    out.insert(name.clone(), Value::Object(d));
                }
            }
        }
        (!out.is_empty()).then_some(Value::Object(out))
    }
    let mut out = serde_json::Map::new();
    if let Some(p) = props(&committed["properties"], &proposed["properties"]) {
        out.insert("properties".into(), p);
        if let Some(value) = committed.get("subobjects") { out.insert("subobjects".into(), value.clone()); }
    }
    if let Some(rt) = proposed.get("runtime").and_then(Value::as_object) {
        let have = committed.get("runtime").and_then(Value::as_object);
        let add: serde_json::Map<String, Value> = rt.iter().filter(|(k, _)| have.is_none_or(|h| !h.contains_key(*k))).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !add.is_empty() {
            out.insert("runtime".into(), Value::Object(add));
        }
    }
    Value::Object(out)
}

/// Parameters learned exactly once from the first indexed value. Copy the
/// committed field definition, never stale proposal parameters, when publishing
/// these additions (dense-vector defaults otherwise become accidental updates).
fn learned_field(have: &Value, proposed: &Value) -> Option<Value> {
    let ty = have.get("type").and_then(Value::as_str)?;
    if proposed.get("type").and_then(Value::as_str) != Some(ty) { return None; }
    if ty == "semantic_text" && have.get("inference_id") != proposed.get("inference_id") { return None; }
    let key = match ty { "dense_vector" => "dims", "semantic_text" => "model_settings", "constant_keyword" => "value", _ => return None };
    if have.get(key).is_some_and(|v| !v.is_null()) { return None; }
    let value = proposed.get(key).filter(|v| !v.is_null())?;
    let mut result = have.as_object()?.clone();
    result.insert(key.into(), value.clone());
    Some(Value::Object(result))
}

fn learned_pending(have: &Value, proposed: &Value) -> bool {
    if learned_field(have, proposed).is_some() { return true; }
    ["properties", "fields"].iter().any(|key| proposed.get(*key).and_then(Value::as_object).is_some_and(|children| {
        children.iter().any(|(name, def)| learned_pending(&have[*key][name], def))
    }))
}

// Dynamic deltas must remain parseable in the committed object's context.
// In particular passthrough requires priority, and flat dotted names cannot
// be reparsed as nested objects. Never copy these from a stale proposal.
fn copy_object_context(committed: &Value, out: &mut serde_json::Map<String, Value>) {
    for key in ["type", "subobjects", "priority", "time_series_dimension"] {
        if let Some(value) = committed.get(key) { out.insert(key.into(), value.clone()); }
    }
}

fn added_properties(local: &Value, proposed: &Value) -> Option<Value> {
    let p = proposed.as_object()?;
    let mut out = serde_json::Map::new();
    for (name, def) in p {
        match local.get(name) {
            None => {
                out.insert(name.clone(), def.clone());
            }
            Some(have) => {
                let mut d = learned_field(have, def).and_then(|v| v.as_object().cloned()).unwrap_or_default();
                for key in ["properties", "fields"] {
                    if let Some(sub) = added_properties(&have[key], &def[key]) { d.insert(key.into(), sub); }
                }
                if !d.is_empty() {
                    copy_object_context(have, &mut d);
                    out.insert(name.clone(), Value::Object(d));
                }
            }
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// Exact parser-derived changes, excluding root metadata. Unlike legacy
/// full-mapping proposals this includes changes to existing field parameters.
pub fn parsed_update(local: &Value, proposed: &Value) -> Option<Value> {
    fn diff(have: &Value, want: &Value) -> Option<Value> {
        let mut out = serde_json::Map::new();
        for (name, def) in want.as_object()? {
            let Some(old) = have.get(name) else { out.insert(name.clone(), def.clone()); continue };
            let mut changes = serde_json::Map::new();
            for (key, value) in def.as_object()? {
                if matches!(key.as_str(), "properties" | "fields") {
                    if let Some(sub) = diff(&old[key], value) { changes.insert(key.clone(), sub); }
                } else if old.get(key) != Some(value) {
                    changes.insert(key.clone(), value.clone());
                }
            }
            if !changes.is_empty() {
                if let Some(ty) = def.get("type") { changes.insert("type".into(), ty.clone()); }
                out.insert(name.clone(), Value::Object(changes));
            }
        }
        (!out.is_empty()).then_some(Value::Object(out))
    }
    let mut out = serde_json::Map::new();
    for key in ["properties", "runtime"] {
        if let Some(delta) = diff(&local[key], &proposed[key]) { out.insert(key.into(), delta); }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// Fill unchanged field parameters from the current committed mapping, not
/// the sender's possibly stale snapshot. Normal mapping merge validates the
/// resulting changes (including races between first values).
pub fn materialize_update(committed: &Value, delta: &Value) -> Value {
    fn fields(have: &Value, delta: &Value) -> Value {
        let mut out = serde_json::Map::new();
        if let Some(defs) = delta.as_object() {
            for (name, change) in defs {
                let mut merged = have.get(name).cloned().unwrap_or_else(|| serde_json::json!({}));
                if let Some(params) = change.as_object() {
                    for (key, value) in params {
                        merged[key] = if matches!(key.as_str(), "properties" | "fields") {
                            fields(&have[name][key], value)
                        } else { value.clone() };
                    }
                }
                out.insert(name.clone(), merged);
            }
        }
        Value::Object(out)
    }
    let mut out = serde_json::Map::new();
    for key in ["properties", "runtime"] {
        if delta.get(key).is_some() { out.insert(key.into(), fields(&committed[key], &delta[key])); }
    }
    // Mapping fragments are parsed standalone. Preserve the committed root
    // interpretation of dotted names without replaying stale proposal settings.
    if out.contains_key("properties") {
        if let Some(subobjects) = committed.get("subobjects") { out.insert("subobjects".into(), subobjects.clone()); }
    }
    Value::Object(out)
}

fn update_applied(have: &Value, delta: &Value) -> bool {
    delta.as_object().is_none_or(|parts| parts.iter().all(|(key, value)| {
        if matches!(key.as_str(), "properties" | "fields" | "runtime") {
            value.as_object().is_none_or(|fields| fields.iter().all(|(name, change)| update_applied(&have[key][name], change)))
        } else {
            have.get(key) == Some(value)
        }
    }))
}

/// Dotted paths of the leaf fields an [`added_mappers`] update adds.
/// Objects (`object`, `nested`, `passthrough`, an untyped def, one with
/// `enabled: false`) are not leaves: the field lookup the wait reads holds
/// leaf fields only, so an empty object (`{"meta": {}}`) would never count
/// as applied.
fn added_paths(update: &Value) -> Vec<String> {
    fn walk(props: &Value, prefix: &str, out: &mut Vec<String>) {
        let Some(p) = props.as_object() else { return };
        for (name, def) in p {
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
            if def.get("properties").is_some() {
                walk(&def["properties"], &path, out);
            } else if def.get("type").and_then(Value::as_str).is_some_and(|t| !matches!(t, "object" | "nested" | "passthrough")) {
                out.push(path);
            }
        }
    }
    // Runtime fields are not waited for (they live outside the field
    // lookup the wait reads).
    let mut out = Vec::new();
    walk(&update["properties"], "", &mut out);
    out
}

/// The local engine copy of `index` with local uuid `uuid`, on any engine.
fn local_copy(node: &ClusterNode, index: &str, uuid: &str) -> Option<Arc<elasticsearch_engine::index::IndexService>> {
    node.all_shard_engines().into_iter().find_map(|(_, engine)| engine.get_index(index).filter(|svc| svc.uuid == uuid))
}

/// The cluster uuid of the local engine copy `local` of `index`, or `None`
/// (the master then merges without the incarnation check). Never waits for
/// the maps: the applier holds them across index deletes, which wait for
/// this write's operation permit.
fn cluster_uuid_of(node: &ClusterNode, index: &str, local: &str) -> Option<String> {
    // The copy knows its incarnation (set when the applier mapped it).
    if let Some(u) = local_copy(node, index, local).and_then(|svc| svc.cluster_uuid().map(str::to_string)) {
        return Some(u);
    }
    let from = |m: &crate::mirror::UuidMap| m.indices.get(index).filter(|(_, l)| l == local).map(|(c, _)| c.clone());
    let lookup = |m: &parking_lot::Mutex<crate::mirror::UuidMap>| m.try_lock_for(Duration::from_millis(20)).and_then(|g| from(&g));
    lookup(&node.uuids).or_else(|| node.shard_engines.all().into_iter().find_map(|(_, se)| lookup(&se.uuids)))
}

/// Whether the engine copy of `index` with local uuid `uuid` is a primary
/// here. Each cluster shard is its own engine index (engine `k` holds shard
/// `k`), so this is the shard copy that parses, not any copy of the index.
/// A copy that is not a primary (a replica applying replicated ops, a
/// translog replay while a copy opens, a recovery) never proposes a
/// mapping: replicas take the primary's (committed) mapping with the
/// operations, and a proposal from the applier's own thread would wait for
/// itself.
fn holds_primary(node: &ClusterNode, index: &str, shard: u32) -> bool {
    // Never the shard slot's mutex: it is held across a promotion, which
    // waits for replica batches in flight (this parse may be one of them).
    node.shards.is_primary(index, shard)
}

/// The engine (= cluster shard) holding exactly the copy `core`, and that
/// copy. Every engine's copy of an index carries the same (cluster) uuid,
/// so only the identity tells them apart.
fn copy_of_core(node: &ClusterNode, core: &elasticsearch_engine::index::IndexCore) -> Option<(u32, Arc<elasticsearch_engine::index::IndexService>)> {
    node.all_shard_engines().into_iter().find_map(|(k, engine)| engine.get_index(&core.name).filter(|svc| std::ptr::eq::<elasticsearch_engine::index::IndexCore>(&***svc, core)).map(|svc| (k, svc)))
}

/// The reason text of a replica operation that met fields whose committed
/// mapping has not arrived (the primary resends the batch with its mapping).
pub const MAPPING_NOT_ARRIVED: &str = "has not reached this copy yet";

/// A non-primary copy meets fields its mapping does not have: the committed
/// mapping comes ahead of the operations (replication requests, recovery
/// batches), so it is missing only while it is on its way. The copy never
/// derives a mapping of its own (ES retries the replica operation, then
/// fails the copy): the operation fails, retryably. A translog replay on
/// the applier thread (opening a copy) derives (its operations' fields
/// were committed before they were indexed).
thread_local! {
    static NO_DERIVED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with no mapping derived on this thread, even on the applier
/// (a promotion applying held ops: their mappings were never committed).
pub fn without_derived_mappings<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            NO_DERIVED.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(NO_DERIVED.with(|c| c.replace(true)));
    f()
}

fn committed_fields_present(svc: &elasticsearch_engine::index::IndexService, index: &str, base: Option<&Value>, mappings: &Value) -> elasticsearch_engine::error::Result<()> {
    if std::thread::current().name() == Some("cluster-applier") && !NO_DERIVED.with(|c| c.get()) {
        return Ok(());
    }
    let local_mapping = svc.mapping_json();
    let Some(update) = parsed_update(base.unwrap_or(&local_mapping), mappings) else { return Ok(()) };
    let paths = added_paths(&update);
    if update_applied(&svc.mapping_json(), &update) {
        return Ok(());
    }
    Err(EsError::new(503, "mapper_exception", format!("the committed mapping for {paths:?} of [{index}] {MAPPING_NOT_ARRIVED}")))
}

/// Ask the master to commit `req`, then wait until this node applied a
/// state at least that new and the local copy has the added fields.
/// A master change or a publication that did not commit is retried until
/// the timeout (ES retries the mapping update through a failover).
async fn commit_and_apply(node: &Arc<ClusterNode>, req: MappingUpdateReq, timeout: Duration, wait_for: (Arc<elasticsearch_engine::index::IndexService>, Vec<String>, Value)) -> Result<(), EsError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let index = req.index.clone();
    let request = UuidMappingUpdate::from_request(&req)
        .map_err(|e| EsError::try_owned(e.status, &e.error_type, &e.reason).unwrap_or_else(|refusal| refusal))?;
    let version = loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        match node.mapping_update_to_master(&req, &request, left).await {
            Ok(v) => break v,
            Err(e) if (master_unavailable(&e) || e.error_type == "failed_to_commit_cluster_state_exception") && tokio::time::Instant::now() + Duration::from_millis(200) < deadline => {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(e) if master_unavailable(&e) || e.error_type == "failed_to_commit_cluster_state_exception" => return Err(timed_out(&index, timeout)),
            Err(e) => return Err(EsError::try_owned(e.status, &e.error_type, &e.reason).unwrap_or_else(|refusal| refusal)),
        }
    };
    let mut rx = node.subscribe();
    loop {
        if rx.borrow_and_update().version >= version {
            break;
        }
        match tokio::time::timeout_at(deadline, rx.changed()).await {
            Ok(Ok(())) => {}
            _ => return Err(apply_timed_out(timeout)),
        }
    }
    // The applied state carries the mapping; the local copy has it once its
    // engine applied it (on the master engine 0 can lag a kept batch).
    // The very copy that asked (another engine's copy of the index may
    // apply the state earlier).
    let (copy, _paths, update) = wait_for;
    loop {
        let ready = copy.is_detached() || {
            update_applied(&copy.mapping_json(), &update)
        };
        if ready {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(apply_timed_out(timeout));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The mapping committed, but this node did not apply it in time (ES:
/// `MapperException` "timed out while waiting for a dynamic mapping
/// update"). 503, so the engine fails only the items that needed it.
fn apply_timed_out(timeout: Duration) -> EsError {
    EsError::new(503, "mapper_exception", format!("timed out while waiting for a dynamic mapping update ({}ms)", timeout.as_millis()))
}

impl elasticsearch_engine::index::MappingUpdateHook for ClusterMappingHook {
    fn mapping_update(&self, index: &str, uuid: &str, mappings: &Value) -> elasticsearch_engine::error::Result<()> {
        // Without the copy's identity: the first engine holding it.
        let Some(node) = self.node.upgrade() else { return Ok(()) };
        let found = node.all_shard_engines().into_iter().find_map(|(k, engine)| engine.get_index(index).filter(|svc| svc.uuid == uuid).map(|svc| (k, svc)));
        self.update(&node, index, uuid, found, None, mappings)
    }

    fn mapping_update_for(&self, core: &elasticsearch_engine::index::IndexCore, mappings: &Value) -> elasticsearch_engine::error::Result<()> {
        let Some(node) = self.node.upgrade() else { return Ok(()) };
        let found = copy_of_core(&node, core);
        self.update(&node, &core.name, &core.uuid, found, None, mappings)
    }
    fn mapping_update_from(&self, core: &elasticsearch_engine::index::IndexCore, base: &Value, mappings: &Value) -> elasticsearch_engine::error::Result<()> {
        let Some(node) = self.node.upgrade() else { return Ok(()) };
        let found = copy_of_core(&node, core);
        self.update(&node, &core.name, &core.uuid, found, Some(base), mappings)
    }

}

impl ClusterMappingHook {
    /// The hook for the copy `found` (engine / cluster shard, service) of
    /// `index`; `None`: the copy left its engine.
    fn update(&self, node: &Arc<ClusterNode>, index: &str, uuid: &str, found: Option<(u32, Arc<elasticsearch_engine::index::IndexService>)>, base: Option<&Value>, mappings: &Value) -> elasticsearch_engine::error::Result<()> {
        // Engine work inside a metadata batch publishes its own mapping
        // changes with the batch: asking the master would wait for itself.
        if crate::master_service::on_metadata_service() {
            return Ok(());
        }
        let node = node.clone();
        // A copy no longer in its engine is being deleted (its fence waits
        // for this write's permit): fail at once.
        let Some((shard, svc)) = found else {
            return Err(EsError::new(503, elasticsearch_engine::index::COPY_REPLACED, format!("the local copy of index [{index}] was removed while it was being written to")));
        };
        if !holds_primary(&node, index, shard) {
            return committed_fields_present(&svc, index, base, mappings);
        }
        let timeout = dynamic_timeout(&node.state());
        let local_mapping = svc.mapping_json();
        let update = parsed_update(base.unwrap_or(&local_mapping), mappings);
        let Some(update) = update else { return Ok(()) };
        let paths = added_paths(&update);
        let mapping_json = update.to_string();
        // Per shard copy: each one waits for its own copy's fields.
        let key = format!("{index}\u{0}{uuid}\u{0}{shard}\u{0}{mapping_json}");
        let started = std::time::Instant::now();
        // Before any wait (see `cluster_uuid_of`).
        let index_uuid = svc.cluster_uuid().map(str::to_string).or_else(|| cluster_uuid_of(&node, index, uuid));
        let local = Some(svc);
        let encoded_request=encode_mapping_request(index,mappings,index_uuid.as_deref(),Some(&mapping_json))?;
        let (tx, rx) = std::sync::mpsc::channel();
        let first = {
            let mut inflight = self.inflight.lock();
            match inflight.get_mut(&key) {
                Some(waiters) => {
                    waiters.push(tx);
                    false
                }
                None => {
                    inflight.insert(key.clone(), vec![tx]);
                    true
                }
            }
        };
        if first {
            let inflight = self.inflight.clone();
            let slots = self.proposing.clone();
            // The whole proposed mapping, not only the additions: a master
            // from before additive proposals (a rolling upgrade) merges it as
            // it always did, and a current master reads it as its additions
            // against the committed mapping (`additive_update`).
            let req=encoded_request;
            let wait_for = (local.clone().expect("the copy"), paths, update);
            let name = index.to_string();
            self.runtime.spawn(async move {
                // Queued for a proposal slot (never rejected), within the
                // dynamic timeout.
                let result = match tokio::time::timeout(timeout, slots.acquire_owned()).await {
                    Ok(Ok(slot)) => {
                        let left = timeout.saturating_sub(started.elapsed());
                        let r = commit_and_apply(&node, req, left, wait_for).await;
                        drop(slot);
                        r
                    }
                    _ => Err(timed_out(&name, timeout)),
                };
                let waiters = inflight.lock().remove(&key).unwrap_or_default();
                for w in waiters {
                    let _ = w.send(result.clone());
                }
            });
        } else { drop(encoded_request); }
        // The shard's operation permit stays held, as by an ES bulk shard
        // request waiting for its mapping; the write thread does not: a
        // spare worker takes its place in the pool while it waits. A copy
        // being deleted (fenced: it waits for this permit) ends the wait.
        let copy = local;
        elasticsearch_http::pools::blocking(|| {
            let deadline = std::time::Instant::now() + timeout + Duration::from_secs(1);
            loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(result) => return result,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(timed_out(index, timeout)),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
                if copy.as_ref().is_some_and(|svc| svc.is_detached()) {
                    return Err(EsError::new(503, elasticsearch_engine::index::COPY_REPLACED, format!("the local copy of index [{index}] was removed while its mapping update was pending")));
                }
                if std::time::Instant::now() >= deadline {
                    return Err(timed_out(index, timeout));
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_deltas_preserve_parameter_initialization_without_stale_settings() {
        let old = serde_json::json!({"dynamic":"true", "properties":{
            "c":{"type":"constant_keyword"}, "v":{"type":"dense_vector"},
            "s":{"type":"semantic_text","inference_id":"model"}}});
        let mut proposed = old.clone();
        proposed["properties"]["c"]["value"] = serde_json::json!("default");
        proposed["properties"]["v"]["dims"] = serde_json::json!(3);
        proposed["properties"]["s"]["model_settings"] = serde_json::json!({"task_type":"text_embedding","dimensions":3});
        let delta = parsed_update(&old, &proposed).unwrap();
        assert!(delta.get("dynamic").is_none());
        assert!(!update_applied(&old, &delta));
        assert!(update_applied(&proposed, &delta));
        let mut current = old.clone();
        current["dynamic"] = serde_json::json!("strict");
        current["properties"]["c"]["meta"] = serde_json::json!({"new":true});
        let merged = materialize_update(&current, &delta);
        assert_eq!(merged["properties"]["c"]["meta"], current["properties"]["c"]["meta"]);
        assert_eq!(merged["properties"]["c"]["value"], "default");
        assert_eq!(merged["properties"]["v"]["dims"], 3);
        assert_eq!(merged["properties"]["s"]["model_settings"], proposed["properties"]["s"]["model_settings"]);
        assert!(parsed_update(&proposed, &proposed).is_none());
    }

    #[test]
    fn exact_parser_delta_keeps_committed_root_subobjects_context() {
        let base = serde_json::json!({"subobjects":false,"properties":{"metric.old":{"type":"long"}}});
        let mut proposed = base.clone();
        proposed["properties"]["metric.new"] = serde_json::json!({"type":"double"});
        let delta = parsed_update(&base, &proposed).unwrap();
        assert!(delta.get("subobjects").is_none(), "wire delta contains only actual changes");
        let materialized = materialize_update(&base, &delta);
        assert_eq!(materialized["subobjects"], false);
        let parsed = elasticsearch_engine::mapping::Mappings::parse(&materialized, &elasticsearch_engine::analysis::AnalysisRegistry::default()).unwrap();
        assert!(parsed.root.properties.contains_key("metric.new"));
        assert!(!parsed.root.properties.contains_key("metric"));
        assert!(materialized["properties"].get("metric.old").is_none());
    }

    #[test]
    fn parser_snapshot_does_not_revert_concurrent_parameter_updates() {
        let base = serde_json::json!({"properties":{"k":{"type":"keyword","ignore_above":256},"c":{"type":"constant_keyword"}}});
        let mut proposed = base.clone();
        proposed["properties"]["c"]["value"] = serde_json::json!("first");
        let delta = parsed_update(&base, &proposed).unwrap();
        assert!(delta["properties"].get("k").is_none());
        let mut current = base.clone();
        current["properties"]["k"]["ignore_above"] = serde_json::json!(512);
        let update = materialize_update(&current, &delta);
        assert!(update["properties"].get("k").is_none());
        assert_eq!(update["properties"]["c"]["value"], "first");
    }

    #[test]
    fn proposals_carry_only_the_added_mappers() {
        let local = serde_json::json!({"dynamic": "strict", "_meta": {"v": 2}, "properties": {"a": {"type": "keyword", "ignore_above": 512},
            "o": {"properties": {"x": {"type": "long"}}}}});
        // Built on a stale local mapping (dynamic true, old ignore_above, old _meta).
        let proposed = serde_json::json!({"dynamic": "true", "_meta": {"v": 1}, "properties": {"a": {"type": "keyword", "ignore_above": 256},
            "o": {"properties": {"x": {"type": "long"}, "y": {"type": "text"}}}, "f": {"type": "long"}}, "runtime": {"r": {"type": "long"}}});
        let update = added_mappers(&local, &proposed).unwrap();
        assert_eq!(update, serde_json::json!({"properties": {"o": {"properties": {"y": {"type": "text"}}}, "f": {"type": "long"}}, "runtime": {"r": {"type": "long"}}}));
        let mut paths = added_paths(&update);
        paths.sort();
        assert_eq!(paths, vec!["f", "o.y"]);
        assert!(added_mappers(&local, &local).is_none());
    }

    #[test]
    fn learned_parameters_are_published_without_reverting_stale_options() {
        let local = serde_json::json!({"properties": {
            "o": {"properties": {"v": {"type": "dense_vector", "element_type": "byte", "index_options": {"type": "hnsw", "m": 32}}}},
            "text": {"type": "text", "fields": {"semantic": {"type": "semantic_text", "inference_id": "model"}}}
        }});
        let mut proposed = local.clone();
        proposed["properties"]["o"]["properties"]["v"]["dims"] = serde_json::json!(3);
        proposed["properties"]["o"]["properties"]["v"]["index_options"]["m"] = serde_json::json!(16);
        let settings = serde_json::json!({"task_type": "text_embedding", "dimensions": 4, "similarity": "cosine", "element_type": "float"});
        proposed["properties"]["text"]["fields"]["semantic"]["model_settings"] = settings.clone();
        let update = added_mappers(&local, &proposed).unwrap();
        assert_eq!(update["properties"]["o"]["properties"]["v"]["dims"], 3);
        assert_eq!(update["properties"]["o"]["properties"]["v"]["index_options"]["m"], 32);
        assert_eq!(update["properties"]["text"]["fields"]["semantic"]["model_settings"], settings);
        assert_eq!(additive_update(&local, &proposed), update);
        assert!(learned_pending(&local, &update));
        let mut committed = local.clone();
        committed["properties"]["o"]["properties"]["v"]["dims"] = serde_json::json!(5);
        committed["properties"]["text"]["fields"]["semantic"]["model_settings"] = settings;
        // A concurrent first document wins. Reparse against its committed
        // dimensions; don't overwrite them or wait forever for our own value.
        assert!(!learned_pending(&committed, &update));
        assert!(added_mappers(&committed, &proposed).is_none());
        assert_eq!(additive_update(&committed, &proposed), serde_json::json!({}));
    }

    #[test]
    fn first_constant_keyword_value_is_committed_before_bulk_reparse() {
        let local = serde_json::json!({"properties": {"data_stream": {"properties": {
            "dataset": {"type": "constant_keyword", "meta": {"owner": "committed"}}
        }}}});
        let mut proposed = local.clone();
        proposed["properties"]["data_stream"]["properties"]["dataset"]["value"] = serde_json::json!("generic");
        proposed["properties"]["data_stream"]["properties"]["dataset"]["meta"] = serde_json::json!({"owner":"stale"});
        let update = added_mappers(&local, &proposed).expect("first value must reach the master");
        let field = &update["properties"]["data_stream"]["properties"]["dataset"];
        assert_eq!(field["value"], "generic");
        assert_eq!(field["meta"]["owner"], "committed");
        assert_eq!(additive_update(&local, &proposed), update);
        assert!(learned_pending(&local, &update), "field existence alone cannot acknowledge learning");
        let mut winner = local.clone();
        winner["properties"]["data_stream"]["properties"]["dataset"]["value"] = serde_json::json!("concurrent");
        assert!(added_mappers(&winner, &proposed).is_none());
        assert_eq!(additive_update(&winner, &proposed), serde_json::json!({}));
        assert!(!learned_pending(&winner, &update), "reparse against the winning value instead of waiting forever");
    }

    #[test]
    fn dynamic_deltas_preserve_committed_flat_and_passthrough_parse_context() {
        use elasticsearch_engine::{analysis::AnalysisRegistry, mapping::Mappings};
        let registry = AnalysisRegistry::default();
        let local = serde_json::json!({"subobjects":false,"properties":{
            "time":{"type":"long"},
            "attributes":{"type":"passthrough","priority":3,"time_series_dimension":true,"properties":{}}
        }});
        let mut proposed = local.clone();
        proposed["subobjects"] = serde_json::json!(true); // stale proposal context cannot win
        proposed["properties"]["attributes"]["priority"] = serde_json::json!(0);
        proposed["properties"]["attributes"]["time_series_dimension"] = serde_json::json!(false);
        proposed["properties"]["time.min"] = serde_json::json!({"type":"long"});
        proposed["properties"]["attributes"]["properties"]["host.name"] = serde_json::json!({"type":"keyword"});
        let delta = added_mappers(&local, &proposed).unwrap();
        assert_eq!(delta["subobjects"], false);
        assert_eq!(delta["properties"]["attributes"]["priority"], 3);
        assert_eq!(delta["properties"]["attributes"]["time_series_dimension"], true);
        assert_eq!(additive_update(&local, &delta), delta);
        let update = Mappings::parse_update(&delta, &registry).unwrap();
        let mut committed = Mappings::parse(&local, &registry).unwrap();
        committed.merge(&update).unwrap();
        let actual = committed.to_json();
        assert_eq!(actual["properties"]["time.min"]["type"], "long");
        assert_eq!(actual["properties"]["attributes"]["properties"]["host.name"]["type"], "keyword");
        assert!(added_mappers(&local, &local).is_none());
    }

    /// Review NH1: objects never count as fields to wait for (the lookup
    /// holds leaves only), so a write adding one does not park.
    #[test]
    fn objects_without_fields_are_not_waited_for() {
        let update = serde_json::json!({"properties": {
            "meta": {"type": "object"},
            "off": {"type": "object", "enabled": false},
            "loose": {"type": "object", "dynamic": false},
            "untyped": {},
            "nest": {"type": "nested"},
            "pass": {"type": "passthrough", "priority": 1},
            "o": {"properties": {"leaf": {"type": "long"}, "inner": {"type": "object"}}},
            "f": {"type": "flattened"}
        }});
        let mut paths = added_paths(&update);
        paths.sort();
        assert_eq!(paths, vec!["f", "o.leaf"]);
        assert!(added_paths(&serde_json::json!({"properties": {"meta": {"type": "object"}}})).is_empty());
    }

    #[test]
    fn the_master_reads_a_stale_full_mapping_as_its_additions() {
        let committed = serde_json::json!({"dynamic": "strict", "properties": {"a": {"type": "keyword", "ignore_above": 512},
            "o": {"dynamic": "strict", "properties": {"x": {"type": "long"}}}}});
        let stale = serde_json::json!({"dynamic": "true", "properties": {"a": {"type": "keyword", "ignore_above": 256},
            "o": {"dynamic": "true", "properties": {"x": {"type": "long"}, "y": {"type": "text"}}}, "f": {"type": "long"}}});
        assert_eq!(additive_update(&committed, &stale), serde_json::json!({"properties": {"o": {"properties": {"y": {"type": "text"}}}, "f": {"type": "long"}}}));
        // A type change survives (the merge rejects it).
        let conflict = serde_json::json!({"properties": {"a": {"type": "long"}}});
        assert_eq!(additive_update(&committed, &conflict), serde_json::json!({"properties": {"a": {"type": "long"}}}));
        assert_eq!(additive_update(&committed, &committed), serde_json::json!({}));
    }

    #[test]
    fn dynamic_timeout_reads_the_cluster_setting() {
        let mut st = elasticsearch_cluster::state::ClusterState::empty("c");
        assert_eq!(dynamic_timeout(&st), DEFAULT_DYNAMIC_TIMEOUT);
        st.metadata.persistent_settings.insert("indices.mapping.dynamic_timeout".into(), Value::String("2s".into()));
        assert_eq!(dynamic_timeout(&st), Duration::from_secs(2));
        st.metadata.transient_settings.insert("indices.mapping.dynamic_timeout".into(), Value::String("500ms".into()));
        assert_eq!(dynamic_timeout(&st), Duration::from_millis(500), "transient wins");
    }

    #[test]
    fn unreachable_master_is_a_timeout_and_rejections_pass_through() {
        assert!(master_unavailable(&RemoteError::new("master_not_discovered_exception", "", 503)));
        assert!(master_unavailable(&RemoteError::new("connect_exception", "x", 500)));
        assert!(!master_unavailable(&RemoteError::new("illegal_argument_exception", "mapper [f] cannot be changed from type [long] to [text]", 400)));
        // The engine's batched path fails only the deferred items on a 503.
        let e = timed_out("i", Duration::from_secs(1));
        assert_eq!((e.status, e.kind.as_str()), (503, "process_cluster_event_timeout_exception"));
    }
}

#[cfg(test)]
mod capability_wire_tests {
    use super::*;

    #[test]
    fn uuid_mapping_action_never_accepts_legacy_or_truncated_payload() {
        #[derive(serde::Serialize)]
        struct OldRequest { index: String, mapping_json: String }
        let old = bincode::serialize(&OldRequest { index: "i".into(), mapping_json: "{}".into() }).unwrap();
        assert!(bincode::deserialize::<UuidMappingUpdate>(&old).is_err());
        let produced=encode_mapping_request("i",&serde_json::json!({}),Some("generation-a"),Some("{\"properties\":{}}")).unwrap();
        let req=UuidMappingUpdate::from_request(&produced).unwrap();
        let bytes = bincode::serialize(&req).unwrap();
        for end in 0..bytes.len() {
            assert!(bincode::deserialize::<UuidMappingUpdate>(&bytes[..end]).is_err(), "truncation at {end}");
        }
        let back: UuidMappingUpdate = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.dynamic_delta, produced.dynamic_delta);
        assert!(back.memory.bytes() >= (back.index.len()+back.mapping_json.len()+back.index_uuid.len()+back.dynamic_delta.as_ref().map_or(0,String::len)) as u64);
        assert_eq!(back.into_request().unwrap().index_uuid.as_deref(), Some("generation-a"));
    }

    #[test]
    fn unfenced_legacy_and_empty_uuid_requests_are_rejected() {
        for uuid in [None, Some(String::new())] {
            let req = MappingUpdateReq::try_new("i","{}",uuid.as_deref()).unwrap();
            assert!(UuidMappingUpdate::from_request(&req).is_err());
            assert!(require_uuid(req.index_uuid.as_deref()).is_err());
        }
        let empty = UuidMappingUpdate { index: "i".into(), mapping_json: "{}".into(), index_uuid: String::new(), dynamic_delta: None, memory:crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &["i","{}"]).unwrap() };
        assert!(empty.into_request().is_err());
    }
}
