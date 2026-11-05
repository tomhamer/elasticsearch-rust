//! Versioned observations for advisory equal-weight placement.
//! Local monotonic age and live process identity are checked on every read.
use crate::front::SharedText;
use crate::{internal_capabilities::{self, Capability}, node::ClusterNode};
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode};
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant}};

mod domain;
pub(crate) mod pressure;
mod cost;
pub(crate) mod lease;

const ACTION: &str = "internal:monitor/capacity/v1";
const TTL: Duration = Duration::from_secs(90);
const FETCH_LIMIT: Duration = Duration::from_secs(6);
const CONCURRENCY: usize = 4;
/// The master collects capacity once every 30 rounds of its 1s background
/// tick (`spawn_background`, `n % 30 == 1`). Skipped ticks, a slow round
/// and the fetch itself make consecutive samples start a little more than
/// this apart.
pub(crate) const CADENCE: Duration = Duration::from_secs(30);
/// Consecutive high samples are continuous evidence of pressure while each
/// started within 2.5 collection periods of the previous one: one late
/// collection keeps the run, a missed one (or more) starts it again.
const CONTINUITY: Duration = Duration::from_millis(CADENCE.as_millis() as u64 * 5 / 2);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Basis { AnonymousResident, AccountedReservations }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Sample {
    version: u8,
    #[serde(deserialize_with="crate::control_memory::decode")]
    node_id: SharedText,
    #[serde(deserialize_with="crate::control_memory::decode")]
    ephemeral_id: SharedText,
    #[serde(deserialize_with="crate::control_memory::decode")]
    transport_address: SharedText,
    sequence: u64,
    pub(crate) process_id: u32,
    pub(crate) basis: Basis,
    pub(crate) effective_budget_bytes: u64,
    pub(crate) measured_bytes: u64,
    /// Overlapping diagnostic categories, NOT additional bytes to sum with RSS.
    pub(crate) reserved_child_bytes: u64,
    pub(crate) shared_admitted_bytes: u64,
}
#[derive(Serialize, Deserialize)]
struct Request { #[serde(deserialize_with="crate::control_memory::decode")] expected_ephemeral_id: SharedText }
#[derive(Clone, Debug, PartialEq, Eq)]
struct Authority { id: SharedText, ephemeral_id: SharedText, term: u64 }
impl Authority {
    fn current(state: &ClusterState) -> Option<Self> {
        let n=state.master()?;
        Some(Self{id:crate::control_memory::text(&n.id).ok()?,ephemeral_id:crate::control_memory::text(&n.ephemeral_id).ok()?,term:state.term()})
    }
}
struct Entry { sample: Sample, started: Instant, domain: Option<domain::Domain>, high_since:Option<Instant>, _memory:Option<elasticsearch_engine::memory::Reservation> }
#[derive(Default)]
pub(crate) struct CapacityCache { authority: Option<Authority>, entries: BTreeMap<String,Entry>, map_memory:Option<elasticsearch_engine::memory::Reservation>, map_slots:usize, last_pressure_attempt:Option<Instant>, pressure_pair_cursor:usize, pressure_shard_cursor:usize }
fn same_process(sample:&Sample, node:&DiscoveryNode)->bool {
    sample.node_id==node.id && sample.ephemeral_id==node.ephemeral_id && sample.transport_address==node.transport_address
}
impl CapacityCache {
    fn observe(&mut self, authority:&Authority, state:&ClusterState, expected:&DiscoveryNode, sample:Sample, started:Instant, now:Instant)->bool {
        self.observe_domain(authority,state,expected,sample,None,started,now)
    }
    fn observe_domain(&mut self, authority:&Authority, state:&ClusterState, expected:&DiscoveryNode, sample:Sample, domain:Option<domain::Domain>, started:Instant, now:Instant)->bool {
        if Authority::current(state).as_ref()!=Some(authority) {return false;}
        if sample.version!=1 || sample.sequence==0 || sample.effective_budget_bytes==0
            || now.checked_duration_since(started).is_none_or(|age|age>FETCH_LIMIT)
            || !same_process(&sample,expected)
            || !state.nodes.get(&expected.id).is_some_and(|n|same_process(&sample,n)) {return false;}
        if self.authority.as_ref()!=Some(authority) {self.entries=BTreeMap::new();self.map_memory=None;self.map_slots=0;self.authority=Some(authority.clone());}
        self.entries.retain(|id,e|state.nodes.get(id).is_some_and(|n|same_process(&e.sample,n))
            && now.checked_duration_since(e.started).is_some_and(|age|age<=TTL));
        if self.entries.get(sample.node_id.as_str()).is_some_and(|old|old.sample.ephemeral_id==sample.ephemeral_id && old.sample.sequence>=sample.sequence) {return false;}
        // Compare collection starts, not this fetch's completion with the
        // previous start: at the real cadence that gap is just over 30s.
        let high_since=if pressure::high(&sample){self.entries.get(sample.node_id.as_str())
            .filter(|old|started.saturating_duration_since(old.started)<=CONTINUITY).and_then(|old|old.high_since).or(Some(started))}else{None};
        // The map can retain its empty root after removal. Keep structural
        // admission with the map itself, not with an individual entry.
        let structural=2*(11*std::mem::size_of::<(String,Entry)>()+20*std::mem::size_of::<usize>());
        if self.map_memory.is_none(){let Ok(m)=elasticsearch_engine::memory::breakers().reserve_native(0,"capacity cache map") else{return false};self.map_memory=Some(m.into_result_owner());}
        if !self.entries.contains_key(sample.node_id.as_str()) && self.entries.len()>=self.map_slots {
            if self.map_memory.as_mut().unwrap().grow(structural as u64,"capacity cache map growth").is_err(){return false;}
            self.map_slots+=1;
        }
        let Ok(memory)=elasticsearch_engine::memory::breakers().reserve_native(sample.node_id.len() as u64,"capacity observation key") else{return false};
        let Ok(key)=crate::mapping_update::copy_wire_text(&sample.node_id) else{return false};
        self.entries.insert(key,Entry{sample,started,domain,high_since,_memory:Some(memory.into_result_owner())});true
    }
    /// Only use a complete set of fresh anonymous-resident observations. The
    /// reserved-allocation fallback has different coverage and is not comparable.
    /// Bound collection skew as well as age; large/partly upgraded clusters fall
    /// back to the allocator's existing tie break rather than favoring unknowns.
    pub(crate) fn placement(&self, state: &ClusterState, now: Instant) -> PlacementSnapshot {
        let samples = self.snapshot(state, now);
        let mut out=PlacementSnapshot::default();let mut bounds:Option<(Instant,Instant)>=None;
        for node in state.data_nodes() {
            let Some(sample)=samples.get(node.id.as_str()) else{return PlacementSnapshot::default()};
            if sample.basis!=Basis::AnonymousResident{return PlacementSnapshot::default()}
            let started=self.entries[&node.id].started;
            if now.checked_duration_since(started).is_none_or(|age|age>Duration::from_secs(30)){return PlacementSnapshot::default()}
            bounds=Some(bounds.map_or((started,started),|(min,max)|(min.min(started),max.max(started))));
            if out.memory.is_none(){let Ok(memory)=elasticsearch_engine::memory::breakers().reserve_native(0,"placement observations") else{return PlacementSnapshot::default()};out.memory=Some(memory.into_result_owner());}
            let Some(bytes)=node.id.len().checked_add(2*(11*std::mem::size_of::<(String,elasticsearch_cluster::allocation::PlacementMemory)>()+20*std::mem::size_of::<usize>())) else{return PlacementSnapshot::default()};
            if out.memory.as_mut().unwrap().grow(bytes as u64,"placement key and map storage").is_err(){return PlacementSnapshot::default()}
            let Ok(key)=crate::mapping_update::copy_wire_text(&node.id) else{return PlacementSnapshot::default()};
            out.values.insert(key,elasticsearch_cluster::allocation::PlacementMemory{measured_bytes:sample.measured_bytes,budget_bytes:sample.effective_budget_bytes});
        }
        if bounds.is_some_and(|(first,last)|last.duration_since(first)>FETCH_LIMIT){return PlacementSnapshot::default()}
        out
    }

    /// Capacity-sensitive movement may require distinct host domains. Never
    /// infer independence from persistent node IDs, transport addresses or PIDs.
    pub(crate) fn independent_hosts(&self, state:&ClusterState, now:Instant, source:&str, target:&str)->bool {
        let samples=self.snapshot(state,now);
        if !samples.contains_key(source) || !samples.contains_key(target) { return false; }
        let (Some(source),Some(target))=(self.entries.get(source),self.entries.get(target)) else{return false};
        if [source,target].iter().any(|e|now.checked_duration_since(e.started).is_none_or(|age|age>Duration::from_secs(30))) {return false;}
        match (&source.domain,&target.domain) { (Some(a),Some(b))=>a.independent_host(b), _=>false }
    }

    /// Missing, expired, old-term and restarted-node samples are unavailable.
    /// Callers must not interpret absence as zero pressure.
    pub(crate) fn snapshot<'a>(&'a self,state:&ClusterState,now:Instant)->SampleSnapshot<'a> {
        let mut out=SampleSnapshot::default();
        if self.authority!=Authority::current(state) {return out;}
        for (id,e) in &self.entries {
            if !now.checked_duration_since(e.started).is_some_and(|age|age<=TTL) || !state.nodes.get(id).is_some_and(|n|same_process(&e.sample,n)){continue}
            let bytes=2*(11*std::mem::size_of::<(&str,&Sample)>()+20*std::mem::size_of::<usize>());
            if out.memory.is_none(){let Ok(memory)=elasticsearch_engine::memory::breakers().reserve_native(0,"capacity snapshot") else{return SampleSnapshot::default()};out.memory=Some(memory);}
            if out.memory.as_mut().unwrap().grow(bytes as u64,"capacity snapshot map").is_err(){return SampleSnapshot::default()}
            out.values.insert(id.as_str(),&e.sample);
        }
        out
    }
}
/// Owner retained by the reroute caller through consumption of the copied map.
#[derive(Default)]
pub(crate) struct PlacementSnapshot{pub(crate) values:BTreeMap<String,elasticsearch_cluster::allocation::PlacementMemory>,memory:Option<elasticsearch_engine::memory::Reservation>}
impl std::ops::Deref for PlacementSnapshot{type Target=BTreeMap<String,elasticsearch_cluster::allocation::PlacementMemory>;fn deref(&self)->&Self::Target{&self.values}}
impl<'a> IntoIterator for &'a PlacementSnapshot{type Item=(&'a String,&'a elasticsearch_cluster::allocation::PlacementMemory);type IntoIter=std::collections::btree_map::Iter<'a,String,elasticsearch_cluster::allocation::PlacementMemory>;fn into_iter(self)->Self::IntoIter{self.values.iter()}}
/// Borrow immutable samples while owning only the temporary index nodes.
#[derive(Default)]
pub(crate) struct SampleSnapshot<'a>{values:BTreeMap<&'a str,&'a Sample>,memory:Option<elasticsearch_engine::memory::Reservation>}
impl<'a> std::ops::Deref for SampleSnapshot<'a>{type Target=BTreeMap<&'a str,&'a Sample>;fn deref(&self)->&Self::Target{&self.values}}
fn measured_memory(reserved:u64)->(Basis,u64) {
    #[cfg(target_os="linux")]
    {
        use std::io::Read;
        // Kernel status is consumed into bounded stack workspace, not an
        // unadmitted read_to_string allocation on every monitoring request.
        let read=(||->Option<u64>{let mut file=std::fs::File::open("/proc/self/status").ok()?;
            let mut bytes=[0u8;16384];let mut used=0;
            loop{if used==bytes.len(){return None}let n=file.read(&mut bytes[used..]).ok()?;if n==0{break}used+=n;}
            std::str::from_utf8(&bytes[..used]).ok()?.lines().find_map(|line|line.strip_prefix("RssAnon:")
                .and_then(|s|s.split_whitespace().next()).and_then(|n|n.parse::<u64>().ok()).and_then(|n|n.checked_mul(1024)))})();
        if let Some(bytes)=read{return (Basis::AnonymousResident,bytes)}
    }
    (Basis::AccountedReservations,reserved)
}
pub(crate) fn register(node:&Arc<ClusterNode>) {
    domain::register(node);
    pressure::register(node);
    let n=node.clone();let sequence=Arc::new(AtomicU64::new(0));
    node.transport.register_handler(ACTION,move |request:Request| {
        let n=n.clone();let sequence=sequence.clone();
        async move {
            if request.expected_ephemeral_id!=n.local.ephemeral_id {return Err(RemoteError::internal("capacity process identity changed"));}
            let seq=sequence.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|s|s.checked_add(1))
                .map_err(|_|RemoteError::internal("capacity sample sequence exhausted"))?+1;
            let reserved=elasticsearch_engine::memory::breakers().accounted();
            let (basis,measured_bytes)=measured_memory(reserved);
            Ok::<_,RemoteError>(Sample{version:1,node_id:crate::control_memory::text(&n.local.id).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?,ephemeral_id:crate::control_memory::text(&n.local.ephemeral_id).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?,
                transport_address:crate::control_memory::text(&n.local.transport_address).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?,sequence:seq,process_id:std::process::id(),basis,
                effective_budget_bytes:elasticsearch_engine::memory::budget(),measured_bytes,reserved_child_bytes:reserved,
                shared_admitted_bytes:elasticsearch_engine::rustic_core::admission::shared().used()})
        }
    });
}
/// One borrowed peer request; recording occurs immediately when this fetch completes,
/// rather than waiting for the slowest peer in its bounded batch.
async fn fetch_and_observe(node: &ClusterNode, authority: &Authority, peer: Option<&DiscoveryNode>) {
    let Some(peer) = peer else { return };
    let started = Instant::now();
    let sample = tokio::time::timeout(FETCH_LIMIT, async {
        if !internal_capabilities::supports(peer, Capability::CapacityV1) { return None; }
        crate::capability_probe::verify(&node.transport, peer, true).await.ok()?;
        let addr = peer.transport_address.parse().ok()?;
        let sample = node.transport.send_request::<_, Sample>(addr, ACTION,
            &Request { expected_ephemeral_id: crate::control_memory::text(&peer.ephemeral_id).ok()? },
            Duration::from_secs(2)).await.ok()?;
        let domain = domain::fetch(node, peer).await;
        Some((sample, domain))
    }).await.ok().flatten();
    if let Some((sample, domain)) = sample {
        let current = node.state();
        node.master.capacity.lock().observe_domain(authority, &current, peer, sample, domain, started, Instant::now());
    }
}

/// Called only from its independently leased off-actor background task.
pub(crate) async fn collect(node: &ClusterNode, state: &ClusterState) {
    let Some(authority) = Authority::current(state) else { return };
    if authority.id != node.local.id || authority.ephemeral_id != node.local.ephemeral_id { return; }
    // Four concrete futures stay in this task. No boxed fanout storage or
    // higher-ranked borrowed async stream closure escapes into the spawned task.
    const { assert!(CONCURRENCY == 4); }
    let mut peers = state.data_nodes();
    while let Some(first) = peers.next() {
        tokio::join!(
            fetch_and_observe(node, &authority, Some(first)),
            fetch_and_observe(node, &authority, peers.next()),
            fetch_and_observe(node, &authority, peers.next()),
            fetch_and_observe(node, &authority, peers.next()),
        );
    }
    if let Err(error) = pressure::attempt(node).await { tracing::warn!(%error, "measured-capacity relocation refused"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture()->(ClusterState,DiscoveryNode,Sample) {
        let mut state=crate::internal_capabilities::tests::lifecycle_state();
        state.nodes.get_mut("n").unwrap().roles.insert(elasticsearch_cluster::state::Role::Data);
        let node=state.nodes["n"].clone();state.master_node_id=Some(node.id.clone());state.coordination.term=7;
        let sample=Sample{version:1,node_id:crate::control_memory::text(&node.id).unwrap(),ephemeral_id:crate::control_memory::text(&node.ephemeral_id).unwrap(),transport_address:crate::control_memory::text(&node.transport_address).unwrap(),sequence:1,process_id:42,
            basis:Basis::AnonymousResident,effective_budget_bytes:100,measured_bytes:120,reserved_child_bytes:30,shared_admitted_bytes:10};
        (state,node,sample)
    }
    #[test]
    fn capacity_identity_sequence_and_expiry_are_fenced_without_clipping_pressure() {
        let (mut state,node,mut sample)=fixture();let a=Authority::current(&state).unwrap();let now=Instant::now();let mut cache=CapacityCache::default();
        assert!(cache.observe(&a,&state,&node,sample.clone(),now,now));
        assert_eq!(cache.snapshot(&state,now)["n"].measured_bytes,120);
        assert!(!cache.observe(&a,&state,&node,sample.clone(),now,now));
        sample.sequence=2;assert!(cache.observe(&a,&state,&node,sample.clone(),now,now));
        assert!(cache.snapshot(&state,now+TTL+Duration::from_millis(1)).is_empty());
        state.nodes.get_mut("n").unwrap().ephemeral_id.push_str("-restart");
        assert!(cache.snapshot(&state,now).is_empty());
        assert!(!cache.observe(&a,&state,&node,sample,now,now));
    }
    #[test]
    fn capacity_late_or_old_master_samples_never_publish() {
        let (mut state,node,mut sample)=fixture();let a=Authority::current(&state).unwrap();let now=Instant::now();let mut cache=CapacityCache::default();
        assert!(!cache.observe(&a,&state,&node,sample.clone(),now,now+FETCH_LIMIT+Duration::from_millis(1)));
        assert!(!cache.observe(&a,&state,&node,sample.clone(),now+Duration::from_secs(1),now));
        sample.version=2;assert!(!cache.observe(&a,&state,&node,sample.clone(),now,now));sample.version=1;
        sample.effective_budget_bytes=0;assert!(!cache.observe(&a,&state,&node,sample.clone(),now,now));sample.effective_budget_bytes=100;
        assert!(cache.observe(&a,&state,&node,sample.clone(),now,now));state.coordination.term+=1;
        assert!(!cache.observe(&a,&state,&node,sample,now,now));assert!(cache.snapshot(&state,now).is_empty());
    }
    #[test]
    fn placement_requires_complete_comparable_fresh_samples() {
        let (mut state, node, sample) = fixture();
        let a = Authority::current(&state).unwrap();
        let now = Instant::now();
        let mut cache = CapacityCache::default();
        assert!(cache.observe(&a, &state, &node, sample.clone(), now, now));
        assert_eq!(cache.placement(&state, now)["n"].measured_bytes, 120);
        assert!(cache.placement(&state, now + Duration::from_secs(31)).is_empty());
        let mut other = node.clone();
        other.id = "other".into(); other.ephemeral_id = "other-eph".into();
        state.nodes.insert(other.id.clone(), other.clone());
        assert!(cache.placement(&state, now).is_empty());
        let mut second = sample.clone();
        second.node_id = crate::control_memory::text(&other.id).unwrap(); second.ephemeral_id = crate::control_memory::text(&other.ephemeral_id).unwrap();
        second.basis = Basis::AccountedReservations;
        assert!(cache.observe(&a, &state, &other, second.clone(), now, now));
        assert!(cache.placement(&state, now).is_empty());
        second.sequence += 1; second.basis = Basis::AnonymousResident;
        assert!(cache.observe(&a, &state, &other, second.clone(), now + Duration::from_secs(7), now + Duration::from_secs(7)));
        assert!(cache.placement(&state, now + Duration::from_secs(7)).is_empty(), "collection skew must disable preference");
        let mut first = sample; first.sequence += 1;
        assert!(cache.observe(&a, &state, &node, first, now + Duration::from_secs(7), now + Duration::from_secs(7)));
        assert_eq!(cache.placement(&state, now + Duration::from_secs(7)).len(), 2);
        state.coordination.term += 1;
        assert!(cache.placement(&state, now + Duration::from_secs(7)).is_empty());
    }

    #[test]
    fn capacity_metric_bases_remain_explicit_and_wire_is_versioned() {
        let (state,node,mut sample)=fixture();let a=Authority::current(&state).unwrap();let now=Instant::now();let mut cache=CapacityCache::default();
        sample.basis=Basis::AccountedReservations;sample.measured_bytes=30;
        assert!(cache.observe(&a,&state,&node,sample.clone(),now,now));assert_eq!(cache.snapshot(&state,now)["n"].basis,Basis::AccountedReservations);
        let request = bincode::serialize(&Request { expected_ephemeral_id: crate::control_memory::text("e").unwrap() }).unwrap();
        assert_eq!(request, [1u64.to_le_bytes().as_slice(), b"e"].concat());
        let bytes=bincode::serialize(&sample).unwrap();let decoded:Sample=bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.version,1);assert_eq!(decoded.reserved_child_bytes,30);assert_eq!(decoded.shared_admitted_bytes,10);
        // Pin the independent v1 action layout, not an append to DiskWire.
        let mut expected=vec![1];
        for text in ["n","e","127.0.0.1:9300"] {expected.extend_from_slice(&(text.len() as u64).to_le_bytes());expected.extend_from_slice(text.as_bytes());}
        expected.extend_from_slice(&1u64.to_le_bytes()); // sequence
        expected.extend_from_slice(&42u32.to_le_bytes()); // process id
        expected.extend_from_slice(&1u32.to_le_bytes()); // accounted basis
        for value in [100u64,30,30,10] {expected.extend_from_slice(&value.to_le_bytes());}
        assert_eq!(bytes,expected,"capacity v1 wire changes require a new action/version");
    }
}
