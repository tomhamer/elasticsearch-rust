//! Capability claims in node attributes are hints until the actual process
//! answers a dedicated action. Old binaries can echo user-configured attrs.
//! Proofs bind to node id, ephemeral id, transport address and local process;
//! a restarted process never inherits a same-node-id proof.
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode};
use elasticsearch_transport::{AdmittedDiscoveryNode, RemoteError, TransportService};
use elasticsearch_engine::{error::EsError, memory::{self,Reservation}};
use std::{collections::BTreeMap, sync::{Arc, OnceLock, Weak}, time::Duration};

pub(crate) const ACTION: &str = "internal:node/capabilities/v1";
const PREFIX: &str = "rustic.internal.";
// Cache allocations have explicit owners; peer entries retain the actual
// admitted immutable response rather than cloning its attributes into a map.
struct Entries<T> { values: Vec<T>, memory: Option<Reservation> }
impl<T> Default for Entries<T> { fn default()->Self { Self{values:Vec::new(),memory:None} } }
impl<T> Entries<T> {
    fn push(&mut self,value:T)->Result<(),EsError> {
        if self.values.len()==self.values.capacity() {
            let count=self.values.len().checked_add(1).ok_or_else(||EsError::internal("capability cache overflow"))?;
            let bytes=count.checked_mul(std::mem::size_of::<T>()).ok_or_else(||EsError::internal("capability cache overflow"))?;
            let next=memory::breakers().reserve_native(bytes as u64,"capability cache slots")?;
            let mut values=Vec::new();
            values.try_reserve_exact(count).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
            values.append(&mut self.values);
            // Old and new capacities overlap until the old vector is dropped.
            self.values=values;
            self.memory=Some(next.into_result_owner());
        }
        self.values.push(value); Ok(())
    }
}
struct Cache {
    cluster: String,
    local: AdmittedDiscoveryNode,
    peers: parking_lot::Mutex<Entries<AdmittedDiscoveryNode>>,
    _memory: Reservation,
}
static PROOFS: OnceLock<parking_lot::Mutex<Entries<Weak<Cache>>>> = OnceLock::new();
fn owners()->&'static parking_lot::Mutex<Entries<Weak<Cache>>> { PROOFS.get_or_init(Default::default) }
fn find_proofs(cluster:&str,local:&DiscoveryNode)->Option<Arc<Cache>> {
    let mut owners=owners().lock();
    owners.values.retain(|owner|owner.strong_count()>0);
    owners.values.iter().filter_map(Weak::upgrade).find(|owner|owner.cluster==cluster && same_process(&owner.local,local))
}
fn proofs(transport:&TransportService)->Option<Arc<Cache>> { find_proofs(transport.cluster_name(),transport.local_node()) }
fn claims(node:&DiscoveryNode)->impl Iterator<Item=(&String,&String)> { node.attributes.iter().filter(|(key,_)|key.starts_with(PREFIX)) }
fn matches_claims(a:&DiscoveryNode,b:&DiscoveryNode)->bool { claims(a).eq(claims(b)) }
fn cached(owner:&Cache,node:&DiscoveryNode)->bool { owner.peers.lock().values.iter().any(|old|same_process(old,node)&&matches_claims(old,node)) }
fn same_process(expected: &DiscoveryNode, actual: &DiscoveryNode) -> bool {
    expected.id == actual.id && expected.ephemeral_id == actual.ephemeral_id && expected.transport_address == actual.transport_address
}

pub(crate) fn register(transport: &TransportService) -> Result<(),Box<dyn std::error::Error+Send+Sync>> {
    let local=transport.admit_discovery_node(transport.local_node())?;
    let memory=memory::breakers().reserve_native((std::mem::size_of::<Cache>()+2*std::mem::size_of::<usize>()+transport.cluster_name().len()) as u64,"capability cache owner")?;
    let mut cluster=String::new();cluster.try_reserve_exact(transport.cluster_name().len())?;cluster.push_str(transport.cluster_name());
    let owner=Arc::new(Cache{cluster,local:local.clone(),peers:Default::default(),_memory:memory.into_result_owner()});
    {
        let mut owners=owners().lock();owners.values.retain(|owner|owner.strong_count()>0);
        owners.push(Arc::downgrade(&owner))?;
    }
    transport.register_handler(ACTION,move |():()| {
        let local=local.clone();let owner=owner.clone();
        async move {let _owner=owner;Ok::<_,RemoteError>(local)}
    });
    Ok(())
}

/// Live callers (feature requests and join/election sources) also check the
/// current handshake. Relayed metadata may reuse a previously established
/// proof, so a temporarily offline member does not block its own removal.
pub(crate) async fn verify(transport: &TransportService, node: &DiscoveryNode, live: bool) -> Result<(), String> {
    if claims(node).next().is_none() { return Ok(()); }
    if same_process(node, transport.local_node()) {
        return if matches_claims(node,transport.local_node()) { Ok(()) } else { Err("local capability claims differ from this process".into()) };
    }
    let owner = proofs(transport).ok_or_else(|| "local capability probe not registered".to_string())?;
    if !live && cached(&owner,node) { return Ok(()); }
    let addr = node.transport_address.parse().map_err(|_| "invalid capability probe address".to_string())?;
    let identity = tokio::time::timeout(Duration::from_secs(2), transport.connect(addr)).await
        .map_err(|_| "capability handshake timed out".to_string())?.map_err(|e| e.to_string())?;
    if !same_process(node, &identity.node) { return Err(format!("capability probe identity changed for [{}]", node.name)); }
    if cached(&owner,node) { return Ok(()); }
    let response = transport.send_request_bytes(addr, ACTION, bytes::Bytes::new(), Duration::from_secs(2)).await
        .map_err(|e| format!("node [{}] has unverified internal capability attributes: {e}", node.name))?;
    let response=transport.decode_discovery_node(&response).map_err(|e|e.to_string())?;
    if !same_process(node, &response) || !matches_claims(&response,node) {
        return Err(format!("node [{}] capability probe does not match advertised process and features", node.name));
    }
    let mut cache = owner.peers.lock();
    // Retain at most the current process proof for each node in this local
    // cluster. Do not evict unrelated active peers: activation is synchronous.
    cache.values.retain(|old|old.id!=node.id);
    cache.push(response).map_err(|e|e.to_string())?;
    Ok(())
}

/// Verify the actual sender independently from relayed membership. Offline
/// metadata is not a proof, but also cannot block a publication removing it.
pub(crate) async fn verify_sender(transport: &TransportService, address: &str) -> Result<(), String> {
    let addr = address.parse().map_err(|_| "invalid coordination source address".to_string())?;
    let identity = tokio::time::timeout(Duration::from_secs(2), transport.connect(addr)).await
        .map_err(|_| "coordination source handshake timed out".to_string())?.map_err(|e| e.to_string())?;
    verify(transport, &identity.node, true).await
}

pub(crate) fn require_publication_proof(transport: &TransportService, state: &ClusterState) -> Result<(), String> {
    let master = state.master().ok_or_else(|| "publication has no master".to_string())?;
    if claims(master).next().is_none() { return Ok(()); }
    if process_verified(transport.cluster_name(), transport.local_node(), master) { Ok(()) }
    else { Err("publication authority has no current-process capability proof".into()) }
}

fn process_verified(cluster:&str,local:&DiscoveryNode,node:&DiscoveryNode)->bool {
    if same_process(node,local) { return matches_claims(node,local); }
    find_proofs(cluster,local).is_some_and(|owner|cached(&owner,node))
}

/// The segment format `node` is trusted to read when the master negotiates
/// the cluster's format floor: its advertised read maximum when that is the
/// baseline or the claim is process-verified (the local process, or a peer
/// whose join/vote was probed live), else the baseline. An unverified peer
/// holds the floor down until a later publication finds its proof.
pub(crate) fn verified_segment_read_max(cluster: &str, local: &DiscoveryNode, node: &DiscoveryNode) -> u32 {
    let read = elasticsearch_cluster::compatibility::advertised_read_max(node);
    if read <= elasticsearch_cluster::compatibility::BASELINE_SEGMENT_FORMAT || process_verified(cluster, local, node) { read }
    else { elasticsearch_cluster::compatibility::BASELINE_SEGMENT_FORMAT }
}

/// A write in a format no newer than the cluster's segment format floor
/// needs no per-peer proof: the floor only rose on process-verified read
/// maxima, and since then no member reading less can join, vote or be
/// published to (`require_segment_floor`, checked against live-probed
/// joiners), so every member reads it. Proofs are demanded only for a write
/// above the floor (an explicit writer ahead of the negotiated format, or a
/// v12 writer before any floor). A restarted node advertises the committed
/// floor it last applied, so a proof this process has not (yet) collected,
/// a probe that timed out during a full-cluster restart, or a crashed peer
/// still named in the state cannot refuse writes in the floor.
pub(crate) fn require_writer_proofs(state: &ClusterState, local: &DiscoveryNode) -> Result<(), String> {
    let write = elasticsearch_cluster::compatibility::segment_formats(local)?.1;
    if write <= elasticsearch_cluster::compatibility::effective_segment_format_floor(&state.metadata) { return Ok(()); }
    for node in state.data_nodes().filter(|node| node.id != local.id) {
        if !process_verified(&state.cluster_name, local, node) {
            return Err(format!("v12 write awaits current-process capability proof for [{}]",node.name));
        }
    }
    Ok(())
}

/// Warm proofs for reachable peers, without granting anything to unverified
/// relayed claims. Feature activation and writes still demand those proofs.
pub(crate) async fn observe_members(transport: &TransportService, nodes: &BTreeMap<String, DiscoveryNode>) {
    observe(transport, nodes.values()).await
}

async fn observe<'a>(transport: &TransportService, mut peers: impl Iterator<Item = &'a DiscoveryNode>) {
    // Bounded inline fan-out: no Vec/FuturesUnordered allocation, while an
    // unreachable peer does not serialize every other probe behind its timer.
    async fn one(transport:&TransportService,node:Option<&DiscoveryNode>) {
        if let Some(node)=node { let _=verify(transport,node,false).await; }
    }
    while let Some(first)=peers.next() {
        tokio::join!(one(transport,Some(first)),one(transport,peers.next()),
            one(transport,peers.next()),one(transport,peers.next()));
    }
}

/// A member making capability claims this process holds no current proof for.
fn unproven(transport: &TransportService, node: &DiscoveryNode) -> bool {
    claims(node).next().is_some() && !process_verified(transport.cluster_name(), transport.local_node(), node)
}

const REPROBE_MIN: Duration = Duration::from_secs(1);
const REPROBE_MAX: Duration = Duration::from_secs(30);

/// `observe_members` only runs for publications that change membership,
/// with a 2 s probe timeout, so a peer that did not answer then (a full
/// cluster restart, a slow start) would stay unproven until membership
/// changed again. Re-probe members of the applied state that still lack a
/// proof, backing off (doubling to 30 s) while the same ones keep failing
/// and resetting once progress is made or nothing is missing. Ends when the
/// node's state channel closes or its probe registration is gone.
pub(crate) async fn reprobe_missing(transport: TransportService, states: tokio::sync::watch::Receiver<std::sync::Arc<ClusterState>>) {
    reprobe_with(transport, states, REPROBE_MIN, REPROBE_MAX).await
}

async fn reprobe_with(transport: TransportService, mut states: tokio::sync::watch::Receiver<std::sync::Arc<ClusterState>>, min: Duration, max: Duration) {
    let mut delay = min;
    loop {
        tokio::time::sleep(delay).await;
        if states.has_changed().is_err() || proofs(&transport).is_none() { return; }
        let state = states.borrow_and_update().clone();
        let missing = || state.nodes.values().filter(|node| unproven(&transport, node)).count();
        let before = missing();
        if before == 0 { delay = min; continue; }
        observe(&transport, state.nodes.values().filter(|node| unproven(&transport, node))).await;
        let after = missing();
        if after > 0 { tracing::debug!(unproven = after, retry_ms = delay.as_millis() as u64, "members still lack a capability proof"); }
        delay = if after < before { min } else { delay.saturating_mul(2).min(max) };
    }
}

/// Synchronous activation fence on the coordinator actor. Membership handling
/// probes asynchronously before enqueueing messages; never perform network I/O
/// while holding up the actor's election/heartbeat progress.
///
/// Once the cluster's segment format floor rose past the baseline, every
/// member reads it and is a capability-advertising build: the floor only
/// rose on proven read maxima, and every later joiner was probed live and
/// checked against it (`require_segment_floor`). Members then need no proof
/// in this process unless some data member writes above the floor, so a new
/// master after a failover, which admitted none of them itself, does not
/// refuse every task until it happens to re-probe them. The attribute
/// checks (`validate_lifecycle_publication`) still apply.
pub(crate) fn require_activation_proofs(transport: &TransportService, state: &ClusterState) -> Result<(), String> {
    use elasticsearch_cluster::compatibility::{effective_segment_format_floor, segment_formats, BASELINE_SEGMENT_FORMAT};
    let floor = effective_segment_format_floor(&state.metadata);
    let v12 = state.data_nodes().any(|n| segment_formats(n).is_ok_and(|(_, write)| write > floor));
    if !v12 && floor > BASELINE_SEGMENT_FORMAT { return Ok(()); }
    if !v12 && !crate::internal_capabilities::has_lifecycle(state) { return Ok(()); }
    for node in state.nodes.values() {
        // The pre-advertisement baseline claims nothing there is to prove and
        // already preserves lifecycle state (internal_capabilities). A v12
        // writer still demands a process proof from every member.
        if !v12 && crate::internal_capabilities::pre_advertisement(node) { continue; }
        let verified = if same_process(node, transport.local_node()) { matches_claims(node,transport.local_node()) }
            else { proofs(transport).is_some_and(|owner| cached(&owner,node)) };
        if !verified { return Err(format!("node [{}] has no process-verified capability advertisement; activation refused", node.name)); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proof_cache_capacity_is_owned_and_refusal_preserves_entries() {
        let budget=memory::Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=memory::RequestScope::new("proof cache",budget);
        let mut entries=Entries::<u64>::default();
        memory::scoped(Some(scope.clone()),||entries.push(7)).unwrap();
        assert_eq!(entries.values,vec![7]);
        assert!(budget.accounting.reserved()>0);
        // Exhaust actual native child admission, not the RSS parent policy.
        budget.accounting.set_limit(1);
        assert!(memory::scoped(Some(scope),||entries.push(9)).is_err());
        assert_eq!(entries.values,vec![7]);
        drop(entries);
        assert_eq!(budget.accounting.reserved(),0);
    }
    #[test]
    fn same_node_id_after_restart_does_not_reuse_capability_identity() {
        let state = crate::internal_capabilities::tests::lifecycle_state();
        let old = &state.nodes["n"];
        let mut restarted = old.clone();
        assert!(same_process(old, &restarted));
        restarted.ephemeral_id.push_str("-restart");
        assert!(!same_process(old, &restarted));
        restarted = old.clone();
        restarted.transport_address = "127.0.0.1:9999".into();
        assert!(!same_process(old, &restarted));
    }
    #[tokio::test]
    async fn configured_attributes_without_the_capability_action_are_not_proof() {
        use elasticsearch_transport::TransportConfig;
        let state = crate::internal_capabilities::tests::lifecycle_state();
        let mut old = state.nodes["n"].clone();
        old.transport_address = "127.0.0.1:0".into();
        old.ephemeral_id = "old-process".into();
        old.roles = elasticsearch_cluster::state::Role::defaults();
        // This listener deliberately has the same attrs a user can configure on
        // an old binary, but no implementation of the capability probe action.
        let peer = TransportService::bind(TransportConfig::new("probe-test", old, "127.0.0.1:0".parse().unwrap())).await.unwrap();
        let mut local = state.nodes["n"].clone();
        local.id = "local".into();
        local.ephemeral_id = "local-process".into();
        local.transport_address = "127.0.0.1:0".into();
        let caller = TransportService::bind(TransportConfig::new("probe-test", local, "127.0.0.1:0".parse().unwrap())).await.unwrap();
        register(&caller).unwrap();
        let mut inherited = crate::internal_capabilities::tests::lifecycle_state();
        inherited.nodes.insert(peer.local_node().id.clone(), peer.local_node().clone());
        assert!(require_activation_proofs(&caller, &inherited).is_err(), "a diff's inherited node attrs are not process proof");
        let result = verify(&caller, peer.local_node(), true).await;
        assert!(result.is_err(), "echoed attrs must not enable a feature");
        register(&peer).unwrap();
        verify(&caller, peer.local_node(), true).await.unwrap();
        assert!(require_activation_proofs(&caller, &inherited).is_ok());
        let mut offline = peer.local_node().clone();
        offline.id = "offline".into();
        offline.ephemeral_id = "offline-process".into();
        offline.transport_address = "127.0.0.1:0".into();
        inherited.master_node_id = Some(peer.local_node().id.clone());
        inherited.nodes.insert(offline.id.clone(),offline);
        observe_members(&caller,&inherited.nodes).await;
        assert!(require_publication_proof(&caller,&inherited).is_ok(), "offline metadata must not prevent authority publication");
        assert!(require_activation_proofs(&caller,&inherited).is_err(), "offline relayed attrs must not grant feature activation");
        inherited.nodes.remove("offline");
        assert!(require_activation_proofs(&caller,&inherited).is_ok());
        let mut persisted = inherited.clone();
        persisted.nodes.get_mut("n").unwrap().ephemeral_id = "previous-process-before-restart".into();
        assert!(verify(&caller, &persisted.nodes["n"], true).await.is_err());
        let mut restarted = peer.local_node().clone();
        restarted.ephemeral_id = "different-process".into();
        assert!(verify(&caller, &restarted, true).await.is_err(), "cached node id must not authorize a different process");
        caller.close().await;
        peer.close().await;
        caller.unregister_handler(ACTION);
        peer.unregister_handler(ACTION);
        assert!(proofs(&caller).is_none(), "weak registry must not retain a removed local process cache");
        assert!(proofs(&peer).is_none());
    }

    fn floor_test_node(id: &str) -> DiscoveryNode {
        let mut node = crate::internal_capabilities::tests::lifecycle_state().nodes["n"].clone();
        node.id = id.into();
        node.name = id.into();
        node.ephemeral_id = format!("{id}-process");
        node.transport_address = "127.0.0.1:0".into();
        node.roles = elasticsearch_cluster::state::Role::defaults();
        node
    }

    fn floor_test_state(cluster: &str, nodes: &[&DiscoveryNode]) -> ClusterState {
        let mut state = crate::internal_capabilities::tests::lifecycle_state();
        state.cluster_name = cluster.into();
        state.metadata.cluster_uuid = "u".into();
        state.nodes.clear();
        for node in nodes { state.nodes.insert(node.id.clone(), (*node).clone()); }
        state.master_node_id = nodes.first().map(|node| node.id.clone());
        state
    }

    /// After the floor reached v13 a restarted node advertises
    /// segment_write=13. A data peer it holds no proof for (its probe timed
    /// out during a full-cluster restart, or it crashed and is still in the
    /// state) must not refuse its writes or a master task (lifecycle state
    /// present); a write above the floor still demands the proof.
    #[tokio::test]
    async fn a_restart_after_the_floor_rose_accepts_writes_without_a_peer_proof() {
        use elasticsearch_cluster::compatibility::{raise_segment_format, SEGMENT_WRITE};
        use elasticsearch_transport::TransportConfig;
        let caller = TransportService::bind(TransportConfig::new("floor-test", floor_test_node("restarted"), "127.0.0.1:0".parse().unwrap())).await.unwrap();
        register(&caller).unwrap();
        let mut writer = caller.local_node().clone();
        writer.attributes.insert(SEGMENT_WRITE.into(), "13".into());
        // Never answered a probe: nothing listens on port 1.
        let mut crashed = floor_test_node("crashed");
        crashed.transport_address = "127.0.0.1:1".into();
        crashed.attributes.insert(SEGMENT_WRITE.into(), "13".into());
        let baseline = floor_test_state("floor-test", &[caller.local_node(), &crashed]);
        observe_members(&caller, &baseline.nodes).await;
        assert!(unproven(&caller, &crashed));
        let err = crate::internal_capabilities::require_local_writer(&baseline, &writer).unwrap_err();
        assert!(err.contains("awaits current-process capability proof"), "{err}");
        assert!(require_activation_proofs(&caller, &baseline).is_err(), "without a floor a v13 writer still needs every proof");

        let mut floor_13 = baseline.clone();
        assert!(raise_segment_format(&mut floor_13, 13));
        crate::internal_capabilities::require_local_writer(&floor_13, &writer).expect("a write in the floor needs no peer proof");
        require_activation_proofs(&caller, &floor_13).expect("master tasks after a failover need no proof for members of the floor");

        // A writer ahead of the floor still waits for every proof.
        let mut floor_12 = baseline.clone();
        assert!(raise_segment_format(&mut floor_12, 12));
        assert!(crate::internal_capabilities::require_local_writer(&floor_12, &writer).is_err());
        assert!(require_activation_proofs(&caller, &floor_12).is_err(), "a data member writing above the floor needs its proof");
        caller.close().await;
        caller.unregister_handler(ACTION);
    }

    /// A peer whose first probe failed (it was still starting) is proven by
    /// the background re-probe without any membership change, after which
    /// writes above the floor proceed; the re-probe ends with the node.
    #[tokio::test]
    async fn background_reprobe_fills_a_missing_proof() {
        use elasticsearch_cluster::compatibility::SEGMENT_WRITE;
        use elasticsearch_transport::TransportConfig;
        let peer = TransportService::bind(TransportConfig::new("reprobe-test", floor_test_node("peer"), "127.0.0.1:0".parse().unwrap())).await.unwrap();
        let caller = TransportService::bind(TransportConfig::new("reprobe-test", floor_test_node("local"), "127.0.0.1:0".parse().unwrap())).await.unwrap();
        register(&caller).unwrap();
        let state = floor_test_state("reprobe-test", &[caller.local_node(), peer.local_node()]);
        let mut writer = caller.local_node().clone();
        writer.attributes.insert(SEGMENT_WRITE.into(), "12".into());
        // The membership-time probe fails: the peer does not answer the action yet.
        observe_members(&caller, &state.nodes).await;
        assert!(unproven(&caller, peer.local_node()));
        assert!(crate::internal_capabilities::require_local_writer(&state, &writer).is_err());

        let (states, receiver) = tokio::sync::watch::channel(std::sync::Arc::new(state.clone()));
        let task = tokio::spawn(reprobe_with(caller.clone(), receiver, Duration::from_millis(20), Duration::from_millis(200)));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(unproven(&caller, peer.local_node()), "an unanswered re-probe grants nothing");
        register(&peer).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while unproven(&caller, peer.local_node()) {
            assert!(std::time::Instant::now() < deadline, "the background re-probe never proved the peer");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        crate::internal_capabilities::require_local_writer(&state, &writer).expect("the re-probed proof admits the write");
        drop(states);
        tokio::time::timeout(Duration::from_secs(5), task).await.expect("the re-probe ends with the node's state channel").unwrap();
        caller.close().await;
        peer.close().await;
        caller.unregister_handler(ACTION);
        peer.unregister_handler(ACTION);
    }

}
