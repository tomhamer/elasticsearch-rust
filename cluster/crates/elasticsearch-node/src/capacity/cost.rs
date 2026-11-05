//! Allocation-bound resource facts. Counts are owned reservation bytes, never
//! a multiplier of store size or a sum of overlapping RSS/cache estimates.
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_engine::rustic_core::admission::{OwnedBytes,Category};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub node_id:String,
    pub ephemeral_id:String,
    pub index:String,
    pub index_uuid:String,
    pub shard:u32,
    pub allocation_id:String,
    pub primary_term:u64,
}
impl Identity {
    pub(super) fn matches(&self,node:&ClusterNode,state:&ClusterState)->bool {
        self.node_id==node.local.id && self.ephemeral_id==node.local.ephemeral_id
            && state.nodes.get(&self.node_id).is_some_and(|n|n.ephemeral_id==self.ephemeral_id)
            && state.metadata.indices.get(&self.index).is_some_and(|m|m.uuid==self.index_uuid && m.primary_term(self.shard)==self.primary_term)
            && state.routing_table.shard_table(&self.index,self.shard).is_some_and(|table|table.copies.iter().any(|r|
                r.node_id.as_deref()==Some(self.node_id.as_str()) && r.allocation_id()==Some(self.allocation_id.as_str())))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Coverage {
    /// Core and engine guard observations. Historical high-water counters do
    /// not bound a different target generation or recovery workspace; this cannot authorize a capacity relocation.
    PartialCoreReservations,
}
#[derive(Clone, Debug)]
pub(crate) struct Cost {
    pub identity:Identity,
    pub current_owned_bytes:u64,
    pub historical_peak_owned_bytes:u64,
    pub coverage:Coverage,
    /// Disjoint reservation categories; these are not added to RSS.
    pub categories:[OwnedBytes;5],
    pub unplanned_allocations:bool,
}
impl Cost {
    fn from_owned(identity:Identity,bytes:OwnedBytes)->Self {
        Self{identity,current_owned_bytes:bytes.current,historical_peak_owned_bytes:bytes.peak,coverage:Coverage::PartialCoreReservations,categories:[OwnedBytes{current:0,peak:0};5],unplanned_allocations:true}
    }
    /// Current observed ownership is a lower bound for ranking a candidate,
    /// not predicted reclaimed RSS or a target recovery size. Target capacity
    /// is enforced by its independent prospective allowance.
    pub(crate) fn relocation_candidate_bytes(&self)->Option<u64>{(self.current_owned_bytes>0).then_some(self.current_owned_bytes)}
}
pub(crate) fn local(node:&ClusterNode,identity:&Identity)->Result<Cost,&'static str> {
    let state=node.state();
    if !identity.matches(node,&state){return Err("shard cost identity changed");}
    let service=node.engine_for(identity.shard).and_then(|engine|engine.get_index(&identity.index)).ok_or("shard cost local service absent")?;
    if service.is_detached() || service.cluster_uuid()!=Some(identity.index_uuid.as_str()){return Err("shard cost index incarnation changed");}
    let shard=service.shard(identity.shard).ok_or("shard cost local copy absent")?;
    if shard.is_closed() || shard.allocation_id!=identity.allocation_id{return Err("shard cost allocation changed");}
    let bytes=shard.shard.admitted_owner.bytes();
    // Reject replacement during sampling, including same-name recreated copies.
    let current=node.engine_for(identity.shard).and_then(|engine|engine.get_index(&identity.index));
    if !identity.matches(node,&node.state()) || current.as_ref().is_none_or(|v|!Arc::ptr_eq(v,&service))
        || service.shard(identity.shard).is_none_or(|v|!Arc::ptr_eq(&v,&shard)) || service.is_detached() || shard.is_closed(){
        return Err("shard cost sample raced replacement");
    }
    let mut cost=Cost::from_owned(identity.clone(),bytes);
    cost.categories=[Category::Core,Category::EngineRequest,Category::Fielddata,Category::Eql,Category::Inference]
        .map(|category|shard.shard.admitted_owner.category_bytes(category));
    cost.unplanned_allocations=shard.shard.admitted_owner.has_unplanned_allocations();
    Ok(cost)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_history_is_not_a_complete_recovery_cost() {
        let identity=Identity{node_id:"n".into(),ephemeral_id:"process".into(),index:"i".into(),index_uuid:"uuid".into(),shard:0,allocation_id:"a".into(),primary_term:3};
        let cost=Cost::from_owned(identity,OwnedBytes{current:1024,peak:4096});
        assert_eq!(cost.current_owned_bytes,1024);
        assert_eq!(cost.historical_peak_owned_bytes,4096);
        assert_eq!(cost.relocation_candidate_bytes(),Some(1024));
    }
}
