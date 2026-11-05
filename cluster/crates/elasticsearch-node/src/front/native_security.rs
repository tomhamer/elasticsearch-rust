//! Native user documents are sequenced and replicated by the security-index primary.
use super::*;
use elasticsearch_engine::{error::EsError, index::WriteReq, shard::WriteCond};

const ACTION: &str = "internal:security/mirror_native_user";
const INDEX: &str = ".security-7";

#[derive(Serialize)]
struct Mutation {
    id: String, doc: Option<String>, marker: String,
    #[serde(skip)]
    memory: crate::replication::decode::DecodeMemory,
}
impl Mutation {
    fn try_copy(&self) -> Result<Self, EsError> {
        let memory = crate::replication::decode::DecodeMemory::reserve_texts(
            elasticsearch_engine::memory::breakers(), &[&self.id, self.doc.as_deref().unwrap_or(""), &self.marker])?;
        Ok(Self { id: self.id.clone(), doc: self.doc.clone(), marker: self.marker.clone(), memory })
    }
}
impl<'de> Deserialize<'de> for Mutation {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{Error, SeqAccess, Visitor};
        use crate::replication::decode::{Text, OptionalText, DecodeMemory};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Mutation;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("native user mutation") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut memory = elasticsearch_engine::memory::breakers().reserve_request(0, "native mutation decode")
                    .map_err(|e| A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let id = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let doc = seq.next_element_seed(OptionalText(&mut memory))?.ok_or_else(|| A::Error::invalid_length(1, &self))?;
                let marker = seq.next_element_seed(Text(&mut memory))?.ok_or_else(|| A::Error::invalid_length(2, &self))?;
                Ok(Mutation { id, doc, marker, memory: DecodeMemory::new(memory) })
            }
        }
        d.deserialize_struct("Mutation", &["id", "doc", "marker"], V)
    }
}

pub(super) fn register(node: &Arc<ClusterNode>) {
    let node = node.clone();
    node.transport.clone().register_handler(ACTION, move |mutation: Mutation| {
        let node = node.clone();
        async move { apply(&node, mutation).await }
    });
}

async fn apply(node: &Arc<ClusterNode>, mutation: Mutation) -> Result<(), RemoteError> {
    let primary = node.shards.primary(INDEX, 0).ok_or_else(|| primary_moved(node, None, "local primary missing"))?;
    let _guard = primary.gate.read().await;
    if !still_primary(node, &primary) || !engine_is(&node.engine, &primary) {
        let error = primary_moved(node, Some(&primary), "primary context or engine identity changed");
        // The bounded retry in `mirror` waits for the state change this
        // causes and re-resolves the primary.
        fail_orphaned_primary(node, &primary);
        return Err(error);
    }
    let service = primary.svc.clone();
    let Mutation { id, doc, marker, memory } = mutation;
    let condition = WriteCond { coordinator_id: Some(marker), is_retry: true, ..Default::default() };
    let operation = match doc {
        Some(doc) => WriteReq::Index { id: Some(id), source: doc, routing: None, cond: condition },
        None => WriteReq::Delete { id, routing: None, cond: condition },
    };
    tokio::task::spawn_blocking(move || {
        let _decoded_owner = memory;
        let result = service.write(vec![operation], true).remove(0)?;
        service.refresh_doc_shard(&result.id, None)?;
        Ok::<_, EsError>(result)
    })
        .await.map_err(|error| RemoteError::internal(error.to_string()))?
        .map_err(|error| RemoteError::new(&error.kind, error.reason, error.status))?;
    let sequence = primary.copy().shard.max_seq_no();
    let mut info = primary.replicate_through(sequence, &TransportNet(node.clone())).await
        .map_err(|error| RemoteError::new("replication_failed_exception", error, 503))?;
    if !still_primary(node, &primary) {
        return Err(RemoteError::new("shard_not_in_primary_mode_exception", "security index primary moved before acknowledgement", 503));
    }
    primary.refresh_replicas(&TransportNet(node.clone()), &mut info).await
        .map_err(|error| RemoteError::new("replication_failed_exception", error, 503))?;
    Ok(())
}

// Preserve the exact failed precondition in the eventual bounded-retry error.
// These fields contain routing/copy identity only, never the user document or
// password. A persistent identity mismatch must not be mistaken for one stale
// routing snapshot and "fixed" by lengthening the retry deadline.
fn primary_moved(node: &Arc<ClusterNode>, held: Option<&Arc<PrimaryCtx>>, detail: &str) -> RemoteError {
    let state = node.state();
    let routed = state.routing_table.shard_table(INDEX, 0).and_then(|table| table.active_primary()).and_then(|primary| primary.node_id.as_deref());
    let local = node.shards.get(INDEX, 0).map(|local| {
        let local = local.lock();
        (local.allocation_id.clone(), matches!(&local.role, crate::shards::Role::Primary(_)))
    });
    RemoteError::new("shard_not_in_primary_mode_exception", format!(
        "security index primary moved: {detail}; state_version={} routed_primary={routed:?} local_node={} local_copy={local:?} held_current={} engine_matches={}",
        state.version, node.local.id, held.is_some_and(|primary| still_primary(node, primary)),
        held.is_some_and(|primary| engine_is(&node.engine, primary))), 503)
}

pub(super) async fn mirror(node: &Arc<ClusterNode>, id: String, doc: Option<Value>) -> Result<(), EsError> {
    if !node.state().metadata.indices.contains_key(INDEX) {
        if doc.is_none() { return Ok(()) }
        if !node.is_master() { return Err(EsError::new(503, "not_master_exception", "native security mutation must run on the master")) }
        let (_, version) = node.mutate_engine_metadata(move |engine| {
            if engine.get_index(INDEX).is_some() { return Ok(()) }
            engine.create_index(INDEX, &json!({"settings":{"index":{"hidden":true,"number_of_shards":1,
                "number_of_replicas":0,"auto_expand_replicas":"0-1"}},"aliases":{".security":{"is_hidden":true}}})).map(|_| ())
        }).await?;
        if !node.wait_applied_everywhere(version, Duration::from_secs(30)).await {
            return Err(EsError::new(503, "timeout_exception", "security index creation was not applied everywhere"));
        }
    }
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len()).ok_or_else(|| std::io::Error::other("native mutation size overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut size = Count(0);
    if let Some(document) = &doc {
        serde_json::to_writer(&mut size, document).map_err(|e| EsError::new(500, "serialization_exception", e.to_string()))?;
    }
    let marker = new_doc_id();
    let memory = elasticsearch_engine::memory::breakers().reserve_request(
        (id.capacity() as u64).saturating_add(size.0 as u64).saturating_add(marker.capacity() as u64), "native mutation producer")?;
    let doc = if let Some(document) = doc {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size.0).map_err(|e| EsError::new(429, "circuit_breaking_exception", e.to_string()))?;
        bytes.resize(size.0, 0);
        let mut writer = std::io::Cursor::new(bytes.as_mut_slice());
        serde_json::to_writer(&mut writer, &document).map_err(|e| EsError::new(500, "serialization_exception", e.to_string()))?;
        if writer.position() != size.0 as u64 { return Err(EsError::new(500, "serialization_exception", "native mutation size changed")); }
        Some(String::from_utf8(bytes).map_err(|e| EsError::new(500, "serialization_exception", e.to_string()))?)
    } else { None };
    let mutation = Mutation { id, doc, marker, memory: crate::replication::decode::DecodeMemory::new(memory) };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    retry_primary_mutation(deadline, || async {
        let Some(owner) = primary_node_of(node, INDEX, 0) else {
            return Err(RemoteError::new("unavailable_shards_exception", "security index primary is not active", 503));
        };
        if owner == node.local.id { apply(node, mutation.try_copy().map_err(|e| RemoteError::new(&e.kind, e.reason, e.status))?).await }
        else if let Some(address) = node.address_of(&owner) {
            node.transport.send_request::<_, ()>(address, ACTION, &mutation, deadline.saturating_duration_since(tokio::time::Instant::now())).await
                .map_err(|error| error.to_remote_error())
        } else { Err(RemoteError::new("node_not_connected_exception", "security index primary disconnected", 503)) }
    }, || wait_state_change(node, deadline)).await
        .map_err(|error| EsError::try_owned(error.status, &error.error_type, &error.reason).unwrap_or_else(|refusal| refusal))
}

/// Re-resolve the primary after a stale routing/publication race. The caller
/// retains one mutation marker across attempts, including acknowledgement
/// races, so a promoted copy can deduplicate an already applied write.
async fn retry_primary_mutation<A, F, W, G>(deadline: tokio::time::Instant, mut attempt: A, mut wait: W) -> Result<(), RemoteError>
where A: FnMut() -> F, F: std::future::Future<Output=Result<(),RemoteError>>,
      W: FnMut() -> G, G: std::future::Future<Output=bool> {
    loop {
        let result = tokio::time::timeout_at(deadline, attempt()).await
            .unwrap_or_else(|_| Err(RemoteError::new("timeout_exception", "timed out writing security index", 503)));
        let error = match result { Ok(()) => return Ok(()), Err(error) => error };
        if !matches!(error.error_type.as_str(), "shard_not_in_primary_mode_exception" | "unavailable_shards_exception" | "node_not_connected_exception")
            || tokio::time::Instant::now() >= deadline || !wait().await {
            return Err(error);
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;


    /// A service replacement and a routing-only update are different events.
    /// Reproduce the former through the real metadata mirror, without relaxing
    /// the old primary's identity fence or silently binding it to an empty copy.
    /// This is an invariant test, not proof that YAML24 took this interleaving.
    #[test]
    fn metadata_mirror_preserves_same_incarnation_and_fences_replacement() {
        use crate::mirror::{self, UuidMap};
        use elasticsearch_cluster::state::Metadata;
        let dir = tempfile::tempdir().unwrap();
        let engine = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        engine.create_index(INDEX, &json!({"settings":{"number_of_shards":1,"number_of_replicas":0}})).unwrap();
        let service = engine.get_index(INDEX).unwrap();
        let primary = PrimaryCtx::new(INDEX, 0, "held-security-primary", 1, service.clone()).unwrap();
        service.write(vec![WriteReq::Index {
            id: Some("existing-user".into()), source: json!({"enabled":true}).to_string(), routing: None, cond: WriteCond::default(),
        }], true).remove(0).unwrap();
        assert!(service.get("existing-user").unwrap().is_some());
        let sequence = primary.copy().shard.max_seq_no();
        let mut uuids = UuidMap::default();
        let before = mirror::extract(&engine, &Metadata::default(), &mut uuids);
        let mut updated = before.clone();
        updated.indices.get_mut(INDEX).unwrap().settings.insert("index.refresh_interval".into(), json!("2s"));
        let report = mirror::apply_changes(&engine, &before, &updated, &mut uuids, false, &Default::default());
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert!(engine_is(&engine, &primary));
        assert!(service.get("existing-user").unwrap().is_some());
        assert_eq!(primary.copy().shard.max_seq_no(), sequence);

        // An authoritative new incarnation may replace the named engine
        // service while an old caller still owns its PrimaryCtx. The context
        // must keep its old service; accepting the new shell would reset its
        // sequencing/history and make a retry acknowledge a different store.
        let mut replacement = updated.clone();
        replacement.indices.get_mut(INDEX).unwrap().uuid = elasticsearch_engine::index::new_uuid();
        let report = mirror::apply_changes(&engine, &updated, &replacement, &mut uuids, false, &Default::default());
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.deleted, vec![INDEX.to_string()]);
        assert_eq!(report.created, vec![INDEX.to_string()]);
        assert!(!engine_is(&engine, &primary));
        assert!(Arc::ptr_eq(&primary.svc, &service));
        assert!(!Arc::ptr_eq(&engine.get_index(INDEX).unwrap(), &service));
        // Retrying unchanged metadata must not make the stale context valid.
        let report = mirror::apply_changes(&engine, &replacement, &replacement, &mut uuids, false, &Default::default());
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert!(!engine_is(&engine, &primary));
    }

    #[tokio::test]
    async fn primary_move_retries_same_mutation_but_terminal_errors_do_not() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let attempts = AtomicUsize::new(0);
        let mutation = Mutation { id: "user".into(), doc: Some("{}".into()), marker: "stable-request".into(), memory: crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[]).unwrap() };
        let seen = std::sync::Mutex::new(Vec::new());
        retry_primary_mutation(tokio::time::Instant::now()+Duration::from_secs(1), || {
            seen.lock().unwrap().push(mutation.marker.clone());
            let n=attempts.fetch_add(1,Ordering::Relaxed);
            async move { if n<2 {Err(RemoteError::new("shard_not_in_primary_mode_exception","moved",503))} else {Ok(())} }
        }, || async {true}).await.unwrap();
        assert_eq!(*seen.lock().unwrap(),vec!["stable-request";3]);
        let attempts=AtomicUsize::new(0);
        let error=retry_primary_mutation(tokio::time::Instant::now()+Duration::from_secs(1),|| {
            attempts.fetch_add(1,Ordering::Relaxed);
            async {Err(RemoteError::new("version_conflict_engine_exception","conflict",409))}
        },||async {panic!("terminal failure must not wait")}).await.unwrap_err();
        assert_eq!(error.status,409);
        assert_eq!(attempts.load(Ordering::Relaxed),1);
        let error=retry_primary_mutation(tokio::time::Instant::now()+Duration::from_secs(1),
            ||async {Err(RemoteError::new("shard_not_in_primary_mode_exception","moved",503))},
            ||async {false}).await.unwrap_err();
        assert_eq!(error.error_type,"shard_not_in_primary_mode_exception");
    }

    #[test]
    fn native_user_payload_round_trips_on_transport() {
        let mutation = Mutation { id: "user-example".into(),
            doc: Some(json!({"enabled":true,"roles":["reader"],"metadata":{"nullable":null}}).to_string()), marker: "request".into(), memory: crate::replication::decode::DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[]).unwrap() };
        let bytes = bincode::serialize(&mutation).unwrap();
        assert_eq!(bytes, bincode::serialize(&(mutation.id.clone(), mutation.doc.clone(), mutation.marker.clone())).unwrap(), "legacy positional wire unchanged");
        let decoded: Mutation = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.id, mutation.id);
        assert_eq!(decoded.doc, mutation.doc);
        assert_eq!(decoded.marker, mutation.marker);
        assert_eq!(decoded.memory.bytes(), (decoded.id.len() + decoded.doc.as_ref().map_or(0, String::len) + decoded.marker.len()) as u64);
        let copied = decoded.try_copy().unwrap();
        drop(decoded);
        assert_eq!(copied.doc, mutation.doc);
        assert_eq!(copied.memory.bytes(), (copied.id.len() + copied.doc.as_ref().map_or(0, String::len) + copied.marker.len()) as u64);
    }
}
