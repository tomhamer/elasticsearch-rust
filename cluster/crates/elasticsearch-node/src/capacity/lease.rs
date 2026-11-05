//! Target-local recovery admission. State/term changes invalidate unused
//! credits; allocation owners retain charged bytes after cancellation.
use super::Authority;
pub(crate) use super::cost::Identity;
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_engine::rustic_core::admission::{self,Budget,Lease,Reservation};
use std::{collections::BTreeMap,sync::Arc,time::{Duration,Instant}};
const MAX_LEASES:usize=128;
const LIFETIME:Duration=Duration::from_secs(60);
// Progress cannot extend an envelope indefinitely. A timed-out allocation is
// failed explicitly and needs a freshly fenced physical attempt and grant,
// never an unreserved retry.
const MAX_EXECUTION:Duration=Duration::from_secs(60*60);

#[derive(Clone,Debug,PartialEq,Eq)]
pub(crate) struct RecoveryPlan {
    retained:u64,
    workspace:u64,
    /// Pinned source recovery session or object-store commit generation. This
    /// binds the allowance, and is not a prediction derived from store bytes.
    source_generation:u64,
    source_node:String,
    source_ephemeral:String,
}
impl RecoveryPlan {
    /// A hard allowance, not a prediction of successful recovery cost. The
    /// target lends at most half its currently free shared budget so unrelated
    /// work retains headroom. Every attributed allocation consumes this credit;
    /// exhaustion or unplanned allocation fails recovery rather than expanding it.
    fn target_envelope(budget:&Arc<Budget>,source_generation:u64,source_node:&str,source_ephemeral:&str)->Result<Self,String> {
        let retained=budget.limit().saturating_sub(budget.used())/2;
        if retained==0 {return Err("no target recovery admission headroom".into());}
        Ok(Self{retained,workspace:0,source_generation,source_node:source_node.into(),source_ephemeral:source_ephemeral.into()})
    }
    fn total(&self)->Result<u64,&'static str>{self.retained.checked_add(self.workspace).ok_or("recovery plan overflow")}
}
struct Entry {
    identity:Identity,
    authority:Authority,
    plan:RecoveryPlan,
    started:Instant,
    expired:bool,
    lease:Arc<Lease>,
    _memory:Reservation,
}
impl Drop for Entry {fn drop(&mut self){self.lease.expire();}}
/// A task retains its exact grant independently of registry reconciliation.
/// Cancellation drops unused credits immediately even if the task never reaches
/// its normal completion/error handler; the registry retains its tombstone.
pub(crate) struct TaskEnvelope {pub epoch:u64,pub identity:Identity,pub lease:Arc<Lease>,_memory:Reservation}
impl Drop for TaskEnvelope {fn drop(&mut self){self.lease.expire();}}
pub(crate) type TaskSlot=Arc<parking_lot::Mutex<Option<TaskEnvelope>>>;
#[derive(Default)]
pub(crate) struct Registry {entries:BTreeMap<u64,Entry>,next_epoch:u64,pending:BTreeMap<String,Pending>}
fn pressure_allowance(budget:&Arc<Budget>,minimum:u64)->Result<u64,String>{
    let allowance=budget.limit().saturating_sub(budget.used())/2;
    if minimum==0||allowance<minimum{Err("target recovery headroom refused".into())}else{Ok(allowance)}
}
struct Pending { identity:Identity,authority:Authority,source_aid:String,source_node:String,source_ephemeral:String,lease:Arc<Lease>,started:Instant,_memory:Reservation }

impl Registry {
    /// Reserve before the master publishes the exact proposed allocation.
    /// It is an enforced allowance, not a promise that recovery will fit.
    pub(crate) fn prepare_pressure(&mut self,node:&ClusterNode,state:&ClusterState,request:&super::pressure::Grant,now:Instant)->Result<(),String>{
        self.reconcile(node,state,now);
        if self.entries.len()+self.pending.len()>=MAX_LEASES{return Err("recovery admission registry full".into());}
        if state.version!=request.version || state.term()!=request.term || state.master_node_id.as_deref()!=Some(request.master.as_ref())
            || node.local.ephemeral_id!=request.target_ephemeral.as_ref(){return Err("pressure admission authority changed".into());}
        let master=state.master().ok_or("pressure admission master absent")?;
        if master.ephemeral_id!=request.master_ephemeral.as_ref(){return Err("pressure admission master process changed".into());}
        let metadata=state.metadata.indices.get(request.index.as_ref()).ok_or("pressure index absent")?;
        if metadata.uuid!=request.index_uuid.as_ref()||metadata.primary_term(request.shard)!=request.primary_term{return Err("pressure index incarnation changed".into());}
        if !state.nodes.get(request.from_node.as_ref()).is_some_and(|source|source.ephemeral_id==request.source_ephemeral.as_ref()){return Err("pressure source process changed".into());}
        let table=state.routing_table.shard_table(&request.index,request.shard).ok_or("pressure shard absent")?;
        if !table.copies.iter().any(|copy|copy.state==elasticsearch_cluster::state::ShardState::Started&&copy.node_id.as_deref()==Some(request.from_node.as_ref())&&copy.allocation_id()==Some(request.source_aid.as_ref()))
            || table.copies.iter().any(|copy|copy.node_id.as_deref()==Some(node.local.id.as_str())){return Err("pressure source/target routing changed".into());}
        if !request.target_aid.starts_with("capacity-")||self.pending.contains_key(request.target_aid.as_ref()){return Err("pressure allocation token reused".into());}
        let budget=admission::shared();let allowance=pressure_allowance(&budget,request.minimum_bytes)?;
        let bytes=request.index.len()+request.index_uuid.len()+request.target_aid.len().saturating_mul(2)+request.source_aid.len()+request.from_node.len()+request.source_ephemeral.len()+node.local.id.len()+node.local.ephemeral_id.len()+master.id.len()+master.ephemeral_id.len();
        let memory=budget.reserve((bytes as u64).saturating_add(2048),"prospective pressure recovery identity").map_err(|e|e.to_string())?;
        let lease=Lease::reserve(budget,allowance,"prospective pressure recovery allowance").map_err(|e|e.to_string())?;
        let identity=Identity{node_id:node.local.id.clone(),ephemeral_id:node.local.ephemeral_id.clone(),index:request.index.to_string(),index_uuid:request.index_uuid.to_string(),shard:request.shard,allocation_id:request.target_aid.to_string(),primary_term:request.primary_term};
        self.pending.insert(request.target_aid.to_string(),Pending{identity,authority:Authority::current(state).ok_or("pressure authority absent")?,source_aid:request.source_aid.to_string(),source_node:request.from_node.to_string(),source_ephemeral:request.source_ephemeral.to_string(),lease,started:now,_memory:memory});Ok(())
    }
    pub(crate) fn reconcile(&mut self,node:&ClusterNode,state:&ClusterState,now:Instant) {
        let authority=Authority::current(state);
        self.pending.retain(|_,pending| {
            let target= pending.identity.matches(node,state);
            let source=state.nodes.get(&pending.source_node).is_some_and(|source|source.ephemeral_id==pending.source_ephemeral)&&state.routing_table.shard_table(&pending.identity.index,pending.identity.shard).is_some_and(|table|table.copies.iter().any(|copy|copy.allocation_id()==Some(pending.source_aid.as_str())&&copy.node_id.as_deref()==Some(pending.source_node.as_str())));
            let lifetime=if target{MAX_EXECUTION}else{LIFETIME};
            if authority.as_ref()!=Some(&pending.authority)||now.saturating_duration_since(pending.started)>=lifetime||!source{pending.lease.expire();}
            // Retain an expired target marker until the physical allocation is
            // gone; expiry cannot turn a pressure move into an ordinary retry.
            target || (source&&pending.lease.is_active())
        });
        self.entries.retain(|_,entry| {
            if !entry.identity.matches(node,state){return false;}
            if authority.as_ref()!=Some(&entry.authority) || !entry.lease.is_active()
                || (!entry.plan.source_node.is_empty() && !state.nodes.get(&entry.plan.source_node).is_some_and(|source|source.ephemeral_id==entry.plan.source_ephemeral))
                || now.checked_duration_since(entry.started).is_none_or(|age|age>=MAX_EXECUTION)
                || now.saturating_duration_since(entry.lease.last_allocation())>=LIFETIME {
                entry.expired=true;entry.lease.expire();
            }
            // Keep a bounded tombstone until the allocation changes. Otherwise
            // a retried pressure recovery could silently become ordinary,
            // unreserved work after its deadline or master term changed.
            true
        });
    }
    /// Called under the node apply lock. An identical retry keeps its original
    /// deadline and epoch; it never refreshes an abandoned plan indefinitely.
    pub(crate) fn grant(&mut self,node:&ClusterNode,state:&ClusterState,identity:&Identity,plan:RecoveryPlan,now:Instant)->Result<u64,String> {
        self.reconcile(node,state,now);
        if !identity.matches(node,state){return Err("recovery lease allocation identity changed".into());}
        let authority=Authority::current(state).ok_or("recovery lease requires elected authority")?;
        self.grant_validated(identity,authority,plan,now,&admission::shared())
    }
    /// Called under apply_lock after the immutable source response is pinned.
    /// A second task cannot borrow a previous task's grant by repeating its id.
    pub(crate) fn grant_envelope(&mut self,node:&ClusterNode,state:&ClusterState,identity:&Identity,source_generation:u64,source_node:&str,now:Instant)->Result<TaskEnvelope,String> {
        self.reconcile(node,state,now);
        if !identity.matches(node,state){return Err("recovery envelope target identity changed".into());}
        if self.entries.values().any(|entry|entry.identity==*identity){return Err("recovery allocation already has an admission attempt".into());}
        if identity.allocation_id.starts_with("capacity-") {
            let pending=self.pending.remove(&identity.allocation_id).ok_or("pressure relocation has no live target admission")?;
            if pending.identity!=*identity || !pending.lease.is_active(){return Err("pressure relocation target admission expired".into());}
            let source=state.nodes.get(source_node).ok_or("pressure recovery primary absent")?;
            let primary=state.routing_table.shard_table(&identity.index,identity.shard).and_then(|table|table.active_primary()).ok_or("pressure recovery primary absent")?;
            if primary.node_id.as_deref()!=Some(source_node){return Err("pressure recovery primary changed".into());}
            let epoch=self.next_epoch.checked_add(1).ok_or("recovery lease epoch exhausted")?;
            let task_memory={let _scope=pending.lease.enter();admission::shared().reserve((std::mem::size_of::<TaskEnvelope>()+identity.node_id.len()+identity.ephemeral_id.len()+identity.index.len()+identity.index_uuid.len()+identity.allocation_id.len()+source.id.len()+source.ephemeral_id.len()) as u64,"pressure recovery task") .map_err(|e|e.to_string())?};
            let plan=RecoveryPlan{retained:pending.lease.available(),workspace:0,source_generation,source_node:source.id.clone(),source_ephemeral:source.ephemeral_id.clone()};
            let lease=pending.lease.clone();
            self.entries.insert(epoch,Entry{identity:pending.identity,authority:pending.authority,plan,started:pending.started,expired:false,lease:lease.clone(),_memory:pending._memory});
            self.next_epoch=epoch;
            return Ok(TaskEnvelope{epoch,identity:identity.clone(),lease,_memory:task_memory});
        }
        let source=state.nodes.get(source_node).ok_or("recovery envelope source absent")?;
        let primary=state.routing_table.shard_table(&identity.index,identity.shard).and_then(|table|table.active_primary()).ok_or("recovery envelope primary absent")?;
        if primary.node_id.as_deref()!=Some(source_node){return Err("recovery envelope primary changed".into());}
        let master=state.master().ok_or("recovery envelope requires elected authority")?;
        let budget=admission::shared();
        // Guard source strings before the plan clones them. The registry's own
        // durable guard separately owns all copied identity/plan fields.
        let _copy=budget.reserve((source.id.len()+source.ephemeral_id.len()+master.id.len()+master.ephemeral_id.len()) as u64,"recovery envelope plan text").map_err(|e|e.to_string())?;
        let authority=Authority::current(state).ok_or("recovery envelope authority changed")?;
        let plan=RecoveryPlan::target_envelope(&budget,source_generation,&source.id,&source.ephemeral_id)?;
        let task_text=identity.node_id.len()+identity.ephemeral_id.len()+identity.index.len()+identity.index_uuid.len()+identity.allocation_id.len();
        let task_memory=budget.reserve((task_text as u64).saturating_add(std::mem::size_of::<TaskEnvelope>() as u64),"recovery task identity").map_err(|e|e.to_string())?;
        let epoch=self.grant_validated(identity,authority,plan,now,&budget)?;
        Ok(TaskEnvelope{epoch,identity:identity.clone(),lease:self.entries[&epoch].lease.clone(),_memory:task_memory})
    }
    fn grant_validated(&mut self,identity:&Identity,authority:Authority,plan:RecoveryPlan,now:Instant,budget:&Arc<Budget>)->Result<u64,String> {
        for (epoch,entry) in &self.entries {
            if entry.identity==*identity {
                if entry.expired || entry.authority!=authority || entry.plan!=plan{return Err("conflicting recovery admission plan".into());}
                return Ok(*epoch);
            }
        }
        if self.entries.len()>=MAX_LEASES{return Err("recovery admission registry full".into());}
        let epoch=self.next_epoch.checked_add(1).ok_or("recovery lease epoch exhausted")?;
        let text=identity.node_id.len()+identity.ephemeral_id.len()+identity.index.len()+identity.index_uuid.len()+identity.allocation_id.len()+authority.id.len()+authority.ephemeral_id.len()+plan.source_node.len()+plan.source_ephemeral.len();
        let memory=budget.reserve((text as u64).saturating_add(2048),"recovery lease registry").map_err(|e|e.to_string())?;
        let lease=Lease::reserve(budget.clone(),plan.total()?,"target recovery allocation plan").map_err(|e|e.to_string())?;
        self.entries.insert(epoch,Entry{identity:identity.clone(),authority,plan,started:now,expired:false,lease,_memory:memory});
        self.next_epoch=epoch;
        Ok(epoch)
    }
    pub(crate) fn claim(&mut self,node:&ClusterNode,state:&ClusterState,epoch:u64,identity:&Identity,now:Instant)->Result<Arc<Lease>,&'static str> {
        self.reconcile(node,state,now);
        let entry=self.entries.get(&epoch).ok_or("recovery admission lease expired")?;
        if entry.expired{return Err("recovery admission lease expired");}
        if entry.identity!=*identity{return Err("recovery admission lease allocation mismatch");}
        Ok(entry.lease.clone())
    }
    /// Membership check only: callers must not deep-copy an unadmitted
    /// identity just to determine whether a physical attempt already exists.
    pub(crate) fn has_allocation(&mut self,node:&ClusterNode,state:&ClusterState,index:&str,shard:u32,aid:&str,now:Instant)->Result<bool,&'static str> {
        self.reconcile(node,state,now);
        let Some(entry)=self.entries.values().find(|entry|entry.identity.index==index && entry.identity.shard==shard && entry.identity.allocation_id==aid) else{return Ok(false)};
        if entry.expired{return Err("recovery allocation admission expired");}
        Ok(true)
    }
    /// Called only after installation and recovery finalization have passed
    /// their existing identity/checkpoint fences, under the node apply lock.
    pub(crate) fn complete(&mut self,node:&ClusterNode,state:&ClusterState,epoch:u64,identity:&Identity,now:Instant)->Result<(),String> {
        let lease=self.claim(node,state,epoch,identity,now).map_err(str::to_owned)?;
        lease.complete().map_err(|e|e.to_string())?;
        self.entries.remove(&epoch);
        Ok(())
    }
    /// The caller holds apply_lock and has detached/fenced the old physical
    /// copy. A new task must negotiate a new grant; escaped old allocations
    /// remain charged and their expired lease cannot grow again.
    pub(crate) fn retire_detached_attempt(&mut self,index:&str,shard:u32,aid:&str)->Result<(),&'static str> {
        let matches=|entry:&Entry|entry.identity.index==index && entry.identity.shard==shard && entry.identity.allocation_id==aid;
        if self.entries.values().any(|entry|matches(entry) && entry.lease.is_active()) {
            return Err("recovery admission still belongs to an active task");
        }
        self.entries.retain(|_,entry|!matches(entry));
        Ok(())
    }
    pub(crate) fn cancel(&mut self,epoch:u64){if let Some(entry)=self.entries.get_mut(&epoch){entry.expired=true;entry.lease.expire();}}
}
#[cfg(test)]
mod tests {
    use super::*;
    fn identity()->Identity{Identity{node_id:"n".into(),ephemeral_id:"process".into(),index:"i".into(),index_uuid:"uuid".into(),shard:0,allocation_id:"aid".into(),primary_term:7}}
    fn authority()->Authority{Authority{id:crate::front::SharedText::try_new("master").unwrap(),ephemeral_id:crate::front::SharedText::try_new("master-process").unwrap(),term:9}}
    #[test]
    fn target_plan_retry_cancel_and_escaped_allocations_are_exactly_charged() {
        let budget=Budget::new(10000);let mut registry=Registry::default();let now=Instant::now();
        let plan=RecoveryPlan{retained:500,workspace:300,source_generation:17,source_node:String::new(),source_ephemeral:String::new()};
        let epoch=registry.grant_validated(&identity(),authority(),plan.clone(),now,&budget).unwrap();
        let total=budget.used();
        assert_eq!(registry.grant_validated(&identity(),authority(),plan.clone(),now+Duration::from_secs(1),&budget).unwrap(),epoch);
        assert_eq!(registry.entries[&epoch].started,now);assert_eq!(budget.used(),total);
        let allocation={let _scope=registry.entries[&epoch].lease.enter();budget.reserve(400,"opened recovery data").unwrap()};
        assert_eq!(budget.used(),total);
        let mut changed=plan;changed.source_generation+=1;
        assert!(registry.grant_validated(&identity(),authority(),changed,now,&budget).is_err());
        registry.cancel(epoch);assert!(registry.entries[&epoch].expired);
        assert_eq!(budget.used(),total-400);
        drop(registry);assert_eq!(budget.used(),400);
        drop(allocation);assert_eq!(budget.used(),0);
    }
    #[test]
    fn refused_target_plan_never_publishes_registry_entry() {
        let budget=Budget::new(3000);let mut registry=Registry::default();
        assert!(registry.grant_validated(&identity(),authority(),RecoveryPlan{retained:3000,workspace:1,source_generation:1,source_node:String::new(),source_ephemeral:String::new()},Instant::now(),&budget).is_err());
        assert!(registry.entries.is_empty());assert_eq!(registry.next_epoch,0);assert_eq!(budget.used(),0);
    }
}

#[cfg(test)]
mod pressure_tests {
    use super::*;
    #[test]
    fn prospective_allowance_counts_other_recoveries_and_never_credits_source(){
        let budget=Budget::new(1000);
        let source=budget.reserve(200,"retained local source").unwrap();
        let first=Lease::reserve(budget.clone(),pressure_allowance(&budget,300).unwrap(),"first target grant").unwrap();
        assert_eq!(budget.used(),600);
        // The first grant is charged before any target allocation is made.
        assert!(pressure_allowance(&budget,300).is_err());
        let escaped={let _scope=first.enter();budget.reserve(100,"target resident allocation").unwrap()};
        first.expire();assert_eq!(budget.used(),300);
        drop(first);assert_eq!(budget.used(),300);
        drop(escaped);assert_eq!(budget.used(),200);
        drop(source);assert_eq!(budget.used(),0);
    }
    #[test]
    fn denied_or_unknown_minimum_does_not_reserve_target_credit(){
        let budget=Budget::new(100);
        assert!(pressure_allowance(&budget,0).is_err());
        assert!(pressure_allowance(&budget,51).is_err());
        assert_eq!(budget.used(),0);
        let held=budget.reserve(100,"busy target").unwrap();
        assert!(pressure_allowance(&budget,1).is_err());
        drop(held);assert_eq!(budget.used(),0);
    }
}
