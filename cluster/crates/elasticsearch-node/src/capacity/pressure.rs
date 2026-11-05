//! Bounded pressure moves. RSS chooses when to try; actual owner bytes rank
//! candidates. A target grant is an allowance, never a predicted recovery size.
use super::*;
use crate::replication::recovery::RecoveryText as Text;
use elasticsearch_cluster::{allocation::{ShardStoreInfo,patch_versions},state::ShardState};
const GRANT:&str="internal:cluster/capacity/prepare/v1";
const COST:&str="internal:monitor/capacity_cost/v1";
const SUSTAIN:Duration=Duration::from_secs(60);
const COOLDOWN:Duration=Duration::from_secs(600);
#[derive(Clone,Serialize,Deserialize)]
pub(crate) struct Grant {
    pub version:u64,pub term:u64,pub master:Text,pub master_ephemeral:Text,
    pub target_ephemeral:Text,pub index:Text,pub index_uuid:Text,pub shard:u32,
    pub primary_term:u64,pub from_node:Text,pub source_ephemeral:Text,pub source_aid:Text,pub target_aid:Text,pub minimum_bytes:u64,
}
#[derive(Serialize,Deserialize)]
struct CostRequest { index:Text,index_uuid:Text,shard:u32,aid:Text,term:u64,ephemeral:Text }
#[derive(Serialize,Deserialize)]
struct CostReply { bytes:u64 }
pub(super) fn register(node:&Arc<ClusterNode>){
    let n=node.clone();node.transport.register_handler(GRANT,move |request:Grant|{let node=n.clone();async move{
        let budget=elasticsearch_engine::memory::budget();let (basis,used)=super::measured_memory(0);
        if basis!=Basis::AnonymousResident || !below(used,budget,60){return Err(RemoteError::internal("pressure target no longer has low resident pressure"));}
        node.prepare_pressure_recovery(&request).map_err(RemoteError::internal)?;
        Ok::<_,RemoteError>(true)
    }});
    let n=node.clone();node.transport.register_handler(COST,move |request:CostRequest|{let node=n.clone();async move{
        if request.ephemeral.as_ref()!=node.local.ephemeral_id{return Err(RemoteError::internal("pressure cost process changed"));}
        let _identity_memory=elasticsearch_engine::rustic_core::admission::shared().reserve((node.local.id.len()+node.local.ephemeral_id.len()+request.index.len()+request.index_uuid.len()+request.aid.len()) as u64*2,"pressure cost identity and sampled copy").map_err(|error| {
            // Admission refusal must keep its typed 429 across transport.
            crate::control_memory::remote_error(&elasticsearch_engine::error::EsError::from(error))
        })?;
        let identity=super::cost::Identity{node_id:node.local.id.clone(),ephemeral_id:node.local.ephemeral_id.clone(),index:request.index.to_string(),index_uuid:request.index_uuid.to_string(),shard:request.shard,allocation_id:request.aid.to_string(),primary_term:request.term};
        let cost=super::cost::local(&node,&identity).map_err(RemoteError::internal)?;
        // Partial owner attribution is useful only to select a nonempty
        // candidate. It is never promoted to predicted reclaim or target fit.
        Ok::<_,RemoteError>(CostReply{bytes:cost.relocation_candidate_bytes().unwrap_or(0)})
    }});
}
fn below(used:u64,budget:u64,percent:u64)->bool{budget>0&&(used as u128)*100<=(budget as u128)*percent as u128}
pub(super) fn high(sample:&Sample)->bool{sample.basis==Basis::AnonymousResident&&!below(sample.measured_bytes,sample.effective_budget_bytes,85)}
fn text(value:&str)->Result<Text,String>{Text::try_new(value).map_err(|error|error.to_string())}
/// The capacity collection task has a single-flight lease. No polling worker
/// overlaps another proposal; the cluster routing fence also rejects any
/// simultaneous automatic/admin relocation.
pub(super) async fn attempt(node:&ClusterNode)->Result<(),String>{
    let state=node.state();let authority=Authority::current(&state).ok_or("no pressure authority")?;
    if authority.id!=node.local.id||authority.ephemeral_id!=node.local.ephemeral_id{return Ok(());}
    if !state.nodes.values().all(|peer|internal_capabilities::supports(peer,Capability::CapacityPressureV1)){return Ok(());}
    let now=Instant::now();
    let bytes=state.nodes.values().try_fold(1024u64,|bytes,node|bytes.checked_add((node.id.len() as u64).saturating_mul(16))).ok_or("pressure candidate metadata overflow")?;
    let _candidate_memory=elasticsearch_engine::rustic_core::admission::shared().reserve(bytes,"pressure candidate metadata").map_err(|e|e.to_string())?;
    let candidates={let mut cache=node.master.capacity.lock();cache.pressure_candidates(&state,now)};
    if candidates.is_empty(){return Ok(());}
    if state.routing_table.all_shards().any(|copy|copy.state!=ShardState::Started){return Ok(());}
    let mut probes=0;
    let info=node.master.cluster_info.lock().clone();let stores=ShardStoreInfo::from_routing(&state);
    for (source,target) in candidates {
        let Some(source_node)=state.nodes.get(&source)else{continue};let Some(target_node)=state.nodes.get(&target)else{continue};
        // Old versions can supply observations but cannot acknowledge this
        // action. A rejected prepare never publishes a relocation.
        let mut best=None;
        let copies=||state.routing_table.all_shards().filter(|copy|copy.node_id.as_deref()==Some(source.as_str())&&copy.state==ShardState::Started);
        let count=copies().count();if count==0{continue;}
        let start={let cache=node.master.capacity.lock();cache.pressure_shard_cursor%count};
        for copy in copies().skip(start).chain(copies().take(start)){
            if probes>=4||now.elapsed()>FETCH_LIMIT{break;}
            probes+=1;
            let Some(aid)=copy.allocation_id()else{continue};let Some(metadata)=state.metadata.indices.get(&copy.index)else{continue};
            // Eligibility precedes ranking: a larger protected shard must not
            // hide a smaller shard which the automatic deciders permit.
            let Some(next)=node.master.allocation.pressure_relocation(&state,&copy.index,copy.shard,aid,&target,crate::master::now_millis(),&info,&stores)else{continue};
            let request=CostRequest{index:text(&copy.index)?,index_uuid:text(&metadata.uuid)?,shard:copy.shard,aid:text(aid)?,term:metadata.primary_term(copy.shard),ephemeral:text(&source_node.ephemeral_id)?};
            let Ok(cost)=node.transport.send_request::<_,CostReply>(source_node.transport_address.parse().map_err(|_|"bad pressure source address")?,COST,&request,Duration::from_secs(2)).await else{continue};
            if cost.bytes==0{continue;}
            if best.as_ref().is_none_or(|(_,bytes,_)|cost.bytes>*bytes){best=Some((copy,cost.bytes,next));}
        }
        if let Some((copy,bytes,mut next))=best {
            let Some(aid)=copy.allocation_id()else{continue};let Some(metadata)=state.metadata.indices.get(&copy.index)else{continue};
            let Some(target_aid)=next.routing_table.shard_table(&copy.index,copy.shard).and_then(|table|table.copies.iter().find(|c|c.node_id.as_deref()==Some(target.as_str())&&c.state==ShardState::Initializing)).and_then(|copy|copy.allocation_id())else{continue};
            let grant=Grant{version:state.version,term:state.term(),master:text(&authority.id)?,master_ephemeral:text(&authority.ephemeral_id)?,target_ephemeral:text(&target_node.ephemeral_id)?,index:text(&copy.index)?,index_uuid:text(&metadata.uuid)?,shard:copy.shard,primary_term:metadata.primary_term(copy.shard),from_node:text(&source)?,source_ephemeral:text(&source_node.ephemeral_id)?,source_aid:text(aid)?,target_aid:text(target_aid)?,minimum_bytes:bytes};
            node.master.capacity.lock().last_pressure_attempt=Some(now);
            let accepted=node.transport.send_request::<_,bool>(target_node.transport_address.parse().map_err(|_|"bad pressure target address")?,GRANT,&grant,Duration::from_secs(2)).await.map_err(|e|e.to_string())?;
            if !accepted{return Err("target pressure admission refused".into());}
            // Set cooldown on attempted publication too: an indeterminate
            // timeout cannot authorize another competing move.
            node.master.capacity.lock().last_pressure_attempt=Some(now);
            let expected=state.clone();patch_versions(&expected,&mut next);
            node.coord().submit(Box::new(move |current:&ClusterState|{
                if now.elapsed()>=Duration::from_secs(30)||current.version!=expected.version||current.term()!=expected.term()||current.master_node_id!=expected.master_node_id||current.nodes!=expected.nodes||current.routing_table!=expected.routing_table{return Err("pressure proposal superseded".into());}
                Ok(next.clone())
            })).await.map_err(|e|e.to_string())?;
            // Publication acknowledgement is not recovery completion. Retained
            // evidence can correlate this exact grant with later STARTED state.
            tracing::info!(event="capacity_pressure_published",index=%grant.index,shard=grant.shard,
                source=%grant.from_node,target=%target,source_aid=%grant.source_aid,target_aid=%grant.target_aid,
                term=grant.term,prepared_version=grant.version,observed_owner_bytes=grant.minimum_bytes,
                "admitted pressure relocation published; physical recovery pending");
            return Ok(());
        }
    }Ok(())
}
/// Nodes whose fresh samples show sustained high pressure: rebalancing
/// moves nothing onto them (`AllocationService` `can_rebalance_to`). The
/// reroute caller retains this owner while it uses the copied set.
#[derive(Default)]
pub(crate) struct PressuredNodes{pub(crate) values:std::collections::BTreeSet<String>,memory:Option<elasticsearch_engine::memory::Reservation>}
impl CapacityCache {
    /// Same evidence as a pressure move's source: a high anonymous-resident
    /// sample whose run has lasted [`SUSTAIN`]. Missing or stale samples
    /// mean no restriction.
    pub(crate) fn sustained_pressure(&self,state:&ClusterState,now:Instant)->PressuredNodes{
        let samples=self.snapshot(state,now);let mut out=PressuredNodes::default();
        for(id,sample)in samples.iter(){
            let Some(entry)=self.entries.get(*id)else{continue};
            if !high(sample)||entry.high_since.is_none_or(|since|now.saturating_duration_since(since)<SUSTAIN){continue;}
            let bytes=(id.len()+2*(11*std::mem::size_of::<String>()+20*std::mem::size_of::<usize>())) as u64;
            match out.memory.as_mut(){
                Some(memory)=>if memory.grow(bytes,"pressured node set").is_err(){return PressuredNodes::default()},
                None=>{let Ok(memory)=elasticsearch_engine::memory::breakers().reserve_native(bytes,"pressured node set")else{return PressuredNodes::default()};out.memory=Some(memory.into_result_owner());}
            }
            let Ok(key)=crate::mapping_update::copy_wire_text(id)else{return PressuredNodes::default()};
            out.values.insert(key);
        }
        out
    }
    fn pressure_candidates(&mut self,state:&ClusterState,now:Instant)->Vec<(String,String)>{
        if self.last_pressure_attempt.is_some_and(|last|now.saturating_duration_since(last)<COOLDOWN){return Vec::new();}
        let samples=self.placement(state,now);if samples.len()<2{return Vec::new();}
        let mut pairs=Vec::new();
        let mut seen=0usize;let start=self.pressure_pair_cursor;
        for(source,entry)in &self.entries{
            if !samples.contains_key(source)||!high(&entry.sample)||entry.high_since.is_none_or(|since|now.saturating_duration_since(since)<SUSTAIN){continue;}
            for(target,sample)in &samples{
                if source!=target&&below(sample.measured_bytes,sample.budget_bytes,60)&&self.independent_hosts(state,now,source,target){if seen>=start&&pairs.len()<8{pairs.push((source.clone(),target.clone()));}seen+=1;}
            }
        }
        if pairs.is_empty()&&seen>0{self.pressure_pair_cursor=0;return self.pressure_candidates(state,now);}
        // Rotate the first pair every collection, even when all cost probes
        // are exhausted by one source. Do not permanently pin the first node.
        self.pressure_pair_cursor=if seen>0{(start+1)%seen}else{0};
        if self.pressure_pair_cursor==0{self.pressure_shard_cursor=self.pressure_shard_cursor.wrapping_add(1);}
        pairs
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture()->(CapacityCache,ClusterState,Instant){
        let mut state=crate::internal_capabilities::tests::lifecycle_state();
        state.nodes.get_mut("n").unwrap().roles.insert(elasticsearch_cluster::state::Role::Data);
        let original=state.nodes.values().next().unwrap().clone();let mut target=original.clone();target.id="target".into();target.ephemeral_id="target-process".into();target.transport_address="127.0.0.1:9301".into();state.nodes.insert(target.id.clone(),target);
        let now=Instant::now();let mut cache=CapacityCache::default();cache.authority=Authority::current(&state);
        for(node,used,boot)in [(state.nodes.get(&original.id).unwrap(),95,"source-boot"),(state.nodes.get("target").unwrap(),30,"target-boot")]{
            cache.entries.insert(node.id.clone(),Entry{sample:Sample{version:1,node_id:crate::front::SharedText::try_new(&node.id).unwrap(),ephemeral_id:crate::front::SharedText::try_new(&node.ephemeral_id).unwrap(),transport_address:crate::front::SharedText::try_new(&node.transport_address).unwrap(),sequence:1,process_id:1,basis:Basis::AnonymousResident,effective_budget_bytes:100,measured_bytes:used,reserved_child_bytes:0,shared_admitted_bytes:0},started:now,domain:Some(domain::Domain{boot_id:crate::front::SharedText::try_new(boot).unwrap(),cgroup_path:crate::front::SharedText::try_new("/").unwrap(),cgroup_device:1,cgroup_inode:1}),high_since:Some(now-SUSTAIN),_memory:None});
        }(cache,state,now)
    }
    #[test]
    fn bounded_windows_eventually_visit_shards_beyond_the_first_four(){
        let(mut cache,state,now)=fixture();let mut visited=[false;13];
        for _ in 0..13{
            assert_eq!(cache.pressure_candidates(&state,now).len(),1);
            let start=cache.pressure_shard_cursor%visited.len();
            for ordinal in (start..visited.len()).chain(0..start).take(4){visited[ordinal]=true;}
        }
        assert!(visited.into_iter().all(|value|value));
    }
    #[test]
    fn bounded_pairs_rotate_even_when_first_pair_consumes_all_probes(){
        let(mut cache,mut state,now)=fixture();
        for ordinal in 0..12{
            let mut peer=state.nodes["target"].clone();peer.id=format!("target-{ordinal}");peer.ephemeral_id=peer.id.clone();
            let sample=cache.entries["target"].sample.clone();
            cache.entries.insert(peer.id.clone(),Entry{sample:Sample{node_id:crate::front::SharedText::try_new(&peer.id).unwrap(),ephemeral_id:crate::front::SharedText::try_new(&peer.ephemeral_id).unwrap(),transport_address:crate::front::SharedText::try_new(&peer.transport_address).unwrap(),..sample},started:now,domain:Some(domain::Domain{boot_id:crate::front::SharedText::try_format(format_args!("boot-{ordinal}")).unwrap(),cgroup_path:crate::front::SharedText::try_new("/").unwrap(),cgroup_device:1,cgroup_inode:1}),high_since:None,_memory:None});
            state.nodes.insert(peer.id.clone(),peer);
        }
        let mut selected=std::collections::BTreeSet::new();
        for _ in 0..13{let pairs=cache.pressure_candidates(&state,now);assert!(pairs.len()<=8);selected.insert(pairs[0].1.clone());}
        assert_eq!(selected.len(),13);
    }
    #[test]
    fn sustained_pressure_requires_fresh_distinct_domains_and_cooldown(){
        let(mut cache,mut state,now)=fixture();assert_eq!(cache.pressure_candidates(&state,now).len(),1);
        cache.last_pressure_attempt=Some(now);assert!(cache.pressure_candidates(&state,now).is_empty());cache.last_pressure_attempt=None;
        cache.entries.get_mut("target").unwrap().domain.as_mut().unwrap().boot_id=crate::front::SharedText::try_new("source-boot").unwrap();assert!(cache.pressure_candidates(&state,now).is_empty());
        cache.entries.get_mut("target").unwrap().domain=None;assert!(cache.pressure_candidates(&state,now).is_empty());
        let(mut cache,_,_)=fixture();assert!(cache.pressure_candidates(&state,now+TTL+Duration::from_secs(1)).is_empty());
        state.coordination.term+=1;assert!(cache.pressure_candidates(&state,now).is_empty());
    }
    /// Review finding 6: real successive samples at the real collection
    /// cadence (a little over 30s apart, each fetch finishing after it
    /// started) must accumulate into sustained pressure. Nothing here sets
    /// `high_since` by hand.
    #[test]
    fn sustained_pressure_is_detected_from_samples_at_the_collection_cadence(){
        let(_,state,t0)=fixture();
        let authority=Authority::current(&state).unwrap();
        let source=state.nodes.values().find(|n|n.id!="target").unwrap().clone();
        let target=state.nodes["target"].clone();
        let text=|value:&str|crate::front::SharedText::try_new(value).unwrap();
        let sample=|node:&elasticsearch_cluster::state::DiscoveryNode,used:u64,sequence:u64|Sample{version:1,node_id:text(&node.id),ephemeral_id:text(&node.ephemeral_id),
            transport_address:text(&node.transport_address),sequence,process_id:1,basis:Basis::AnonymousResident,effective_budget_bytes:100,measured_bytes:used,reserved_child_bytes:0,shared_admitted_bytes:0};
        let domain=|boot:&str|Some(domain::Domain{boot_id:text(boot),cgroup_path:text("/"),cgroup_device:1,cgroup_inode:1});
        let ms=Duration::from_millis;
        let mut cache=CapacityCache::default();
        let observe=|cache:&mut CapacityCache,started:Instant,sequence:u64|{
            // Each fetch completes a few hundred milliseconds after it starts.
            assert!(cache.observe_domain(&authority,&state,&source,sample(&source,95,sequence),domain("source-boot"),started,started+ms(400)));
            assert!(cache.observe_domain(&authority,&state,&target,sample(&target,30,sequence),domain("target-boot"),started+ms(100),started+ms(500)));
            started+ms(500)
        };
        let period=CADENCE+ms(1300);
        let mut last=t0;
        for round in 0..2u32 {
            last=observe(&mut cache,t0+period*round,u64::from(round)+1);
            assert_eq!(cache.entries[&source.id].high_since,Some(t0),"round {round} continues the run");
        }
        assert!(last.duration_since(t0)<SUSTAIN,"not yet sustained");
        assert!(cache.pressure_candidates(&state,last).is_empty());
        last=observe(&mut cache,t0+period*2,3);
        assert!(last.duration_since(t0)>=SUSTAIN);
        assert_eq!(cache.pressure_candidates(&state,last),vec![(source.id.clone(),target.id.clone())],"sustained pressure is detected");
        assert_eq!(cache.sustained_pressure(&state,last).values.into_iter().collect::<Vec<_>>(),vec![source.id.clone()],"rebalancing avoids the hot node");
        // A missed collection (a gap beyond 2.5 periods) starts a new run.
        let resumed=t0+period*2+CADENCE*5/2+ms(1000);
        observe(&mut cache,resumed,4);
        assert_eq!(cache.entries[&source.id].high_since,Some(resumed));
    }
    #[test]
    fn transient_or_incomplete_observations_do_not_move_shards(){
        let(mut cache,state,now)=fixture();for entry in cache.entries.values_mut(){entry.high_since=Some(now);}
        assert!(cache.pressure_candidates(&state,now).is_empty());
        let(mut cache,state,now)=fixture();cache.entries.remove("target");assert!(cache.pressure_candidates(&state,now).is_empty());
        let(mut cache,state,now)=fixture();cache.entries.get_mut("target").unwrap().sample.basis=Basis::AccountedReservations;assert!(cache.pressure_candidates(&state,now).is_empty());
    }
}
