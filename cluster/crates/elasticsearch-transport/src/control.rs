//! Retained admission for transport-owned control payloads.
//!
//! The fixed-integer bincode schema is walked without allocation before serde
//! materializes strings, tree nodes, boxes or Arc payloads. This is a structural
//! allocation bound, never a multiplier of the encoded frame length.
use crate::{frame::FrameAdmission, RemoteError, TransportError};
use std::sync::Arc;

pub(crate) struct Owner {
    _reservation: Option<Box<dyn Send + Sync>>,
    pub admission: Option<FrameAdmission>,
}
impl std::fmt::Debug for Owner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("ControlOwner") }
}
pub(crate) fn own(bytes: usize, admission: Option<&FrameAdmission>) -> Result<Arc<Owner>, TransportError> {
    let bytes = bytes.checked_add(std::mem::size_of::<Owner>() + 2 * std::mem::size_of::<usize>())
        .ok_or_else(|| TransportError::Serialization("control allocation overflow".into()))?;
    let reservation = admission.map(|a| a(bytes)).transpose().map_err(TransportError::Remote)?;
    Ok(Arc::new(Owner { _reservation: reservation, admission: admission.cloned() }))
}
// Rust's BTree uses B=6: a node holds up to 11 entries and 12 child edges.
// Charge a complete internal node per insertion, including a split's new root.
// The extra 8 words cover parent/index/length/alignment metadata on both nodes.
// This deliberately exceeds packed occupancy, and is independent of string
// lengths (which are counted separately). Update this bound if std changes B.
pub(crate) fn tree_slot<K, V>() -> usize {
    2 * (11 * (std::mem::size_of::<K>() + std::mem::size_of::<V>()) + 20 * std::mem::size_of::<usize>())
}
fn bad() -> TransportError { TransportError::Serialization("invalid or oversized control payload".into()) }
struct Scan<'a> { input: &'a [u8], offset: usize, bytes: usize }
impl<'a> Scan<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], TransportError> {
        let end = self.offset.checked_add(n).ok_or_else(bad)?;
        let data = self.input.get(self.offset..end).ok_or_else(bad)?;
        self.offset = end; Ok(data)
    }
    fn charge(&mut self, n: usize) -> Result<(), TransportError> {
        self.bytes = self.bytes.checked_add(n).ok_or_else(bad)?; Ok(())
    }
    fn count(&mut self) -> Result<usize, TransportError> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap())).map_err(|_|bad())
    }
    fn text(&mut self) -> Result<(), TransportError> {
        let n = self.count()?;
        std::str::from_utf8(self.take(n)?).map_err(|_|bad())?;
        self.charge(n)
    }
    fn map(&mut self) -> Result<(), TransportError> {
        let count = self.count()?;
        // At least two u64 string lengths must remain per map entry.
        if count > (self.input.len() - self.offset) / 16 { return Err(bad()); }
        self.charge(count.checked_mul(tree_slot::<String,String>()).ok_or_else(bad)?)?;
        for _ in 0..count { self.text()?; self.text()?; }
        Ok(())
    }
    fn node(&mut self)->Result<(),TransportError>{
    for _ in 0..7 { self.text()?; } // DiscoveryNode id through ip
    let roles = self.count()?;
    if roles > (self.input.len() - self.offset) / 4 { return Err(bad()); }
    self.charge(roles.checked_mul(tree_slot::<elasticsearch_cluster::state::Role, ()>()).ok_or_else(bad)?)?;
    for _ in 0..roles {
        if u32::from_le_bytes(self.take(4)?.try_into().unwrap()) >= 14 { return Err(bad()); }
    }
    self.map()?; self.text()?; // attributes and version
        Ok(())
    }
    fn error(&mut self, depth: usize) -> Result<(), TransportError> {
        // Bound recursive serde/Drop work before entering serde. Valid deeper
        // errors are rejected as malformed control messages, never truncated.
        if depth >= 128 { return Err(bad()); }
        self.charge(std::mem::size_of::<crate::error::RemoteErrorOwner>() + 2 * std::mem::size_of::<usize>())?;
        self.text()?; self.text()?; self.take(2)?; self.map()?;
        match self.take(1)?[0] {
            0 => Ok(()),
            1 => { self.charge(std::mem::size_of::<RemoteError>())?; self.error(depth + 1) }
            _ => Err(bad()),
        }
    }
}
pub(crate) fn decode_error(payload: &[u8], admission: Option<&FrameAdmission>) -> Result<RemoteError, TransportError> {
    let mut scan = Scan { input: payload, offset: 0, bytes: 0 };
    scan.error(0)?;
    let owner = own(scan.bytes, admission)?;
    let mut value: RemoteError = bincode::deserialize(payload).map_err(|e|TransportError::Serialization(e.to_string()))?;
    value.attach_owner(&owner);
    Ok(value)
}
pub(crate) fn decode_identity(payload: &[u8], admission: Option<&FrameAdmission>) -> Result<crate::NodeIdentity, TransportError> {
    let mut scan = Scan { input: payload, offset: 0, bytes: std::mem::size_of::<crate::service::IdentityOwner>() + 2 * std::mem::size_of::<usize>() };
    scan.text()?; // cluster name
    scan.node()?;
    let owner = own(scan.bytes, admission)?;
    let data = bincode::deserialize(payload).map_err(|e|TransportError::Serialization(e.to_string()))?;
    Ok(crate::NodeIdentity::from_data(data, Some(owner)))
}

pub(crate) fn copy_node(node:&elasticsearch_cluster::state::DiscoveryNode,admission:Option<&FrameAdmission>)->Result<crate::AdmittedDiscoveryNode,TransportError>{
    let mut bytes=std::mem::size_of::<crate::service::DiscoveryNodeOwner>()+2*std::mem::size_of::<usize>();
    for text in [&node.id,&node.name,&node.ephemeral_id,&node.transport_address,&node.http_address,&node.host,&node.ip,&node.version]{bytes=bytes.checked_add(text.len()).ok_or_else(bad)?;}
    bytes=bytes.checked_add(node.roles.len().checked_mul(tree_slot::<elasticsearch_cluster::state::Role,()>()).ok_or_else(bad)?).ok_or_else(bad)?;
    for(k,v)in &node.attributes{bytes=bytes.checked_add(tree_slot::<String,String>()).and_then(|n|n.checked_add(k.len())).and_then(|n|n.checked_add(v.len())).ok_or_else(bad)?;}
    let owner=own(bytes,admission)?;
    Ok(crate::AdmittedDiscoveryNode::new(node.clone(),owner))
}

pub(crate) fn decode_node(payload:&[u8],admission:Option<&FrameAdmission>)->Result<crate::AdmittedDiscoveryNode,TransportError>{
    let mut scan=Scan{input:payload,offset:0,bytes:std::mem::size_of::<crate::service::DiscoveryNodeOwner>()+2*std::mem::size_of::<usize>()};
    scan.node()?;
    let owner=own(scan.bytes,admission)?;
    let node=bincode::deserialize(payload).map_err(|e|TransportError::Serialization(e.to_string()))?;
    Ok(crate::AdmittedDiscoveryNode::new(node,owner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    fn budget(limit: usize) -> (FrameAdmission, Arc<AtomicUsize>) {
        let used = Arc::new(AtomicUsize::new(0));
        struct Charge(usize, Arc<AtomicUsize>);
        impl Drop for Charge { fn drop(&mut self) { self.1.fetch_sub(self.0,SeqCst); } }
        let counter=used.clone();
        let callback: FrameAdmission = Arc::new(move |bytes| {
            if counter.fetch_update(SeqCst,SeqCst,|n|n.checked_add(bytes).filter(|n|*n<=limit)).is_err() {
                return Err(RemoteError::new("circuit_breaking_exception","control fixture limit",429));
            }
            Ok(Box::new(Charge(bytes,counter.clone())))
        });
        (callback,used)
    }
    #[derive(serde::Serialize,serde::Deserialize)]
    struct LegacyError {
        error_type:String, reason:String,status:u16,
        metadata:std::collections::BTreeMap<String,String>,caused_by:Option<Box<LegacyError>>,
    }
    #[test]
    fn recursive_error_wire_and_clones_retain_actual_owner() {
        let legacy=LegacyError {error_type:"outer".into(),reason:"x".repeat(4096),status:400,
            metadata:std::collections::BTreeMap::from([("index".into(),"λ".repeat(100))]),
            caused_by:Some(Box::new(LegacyError{error_type:"inner".into(),reason:"cause".into(),status:500,metadata:Default::default(),caused_by:None}))};
        let wire=bincode::serialize(&legacy).unwrap();
        let (admit,used)=budget(1<<20);
        let value=decode_error(&wire,Some(&admit)).unwrap();
        assert_eq!(bincode::serialize(&value).unwrap(),wire);
        let clone=value.clone();let cause=value.caused_by.as_ref().unwrap().as_ref().clone();
        let mut expected=value.to_json();
        expected.as_object_mut().unwrap().insert("root_cause".into(),serde_json::json!([value.root_cause().to_json()]));
        assert_eq!(serde_json::to_value(value.json_with_root()).unwrap(),expected);
        let held=used.load(SeqCst);assert!(held>=4096);
        drop(wire);drop(value);drop(clone);
        assert_eq!(used.load(SeqCst),held,"independent cause retains shared decode owner");
        assert_eq!(cause.reason,"cause");drop(cause);assert_eq!(used.load(SeqCst),0);
        let (tiny,used)=budget(32);
        assert!(matches!(decode_error(&bincode::serialize(&legacy).unwrap(),Some(&tiny)),Err(TransportError::Remote(ref e)) if e.status==429));
        assert_eq!(used.load(SeqCst),0);
    }
    #[test]
    fn identity_wire_clone_lifetime_and_corrupt_lengths() {
        use elasticsearch_cluster::state::{DiscoveryNode,Role};
        let data=crate::service::NodeIdentityData {cluster_name:"test".into(),node:DiscoveryNode {
            id:"id".into(),name:"node".into(),ephemeral_id:"ephemeral".into(),transport_address:"127.0.0.1:9300".into(),http_address:"127.0.0.1:9200".into(),
            host:"host".into(),ip:"127.0.0.1".into(),roles:Role::defaults(),attributes:std::collections::BTreeMap::from([("zone".into(),"x".repeat(4096))]),version:"9.5.5".into(),
        }};
        let wire=bincode::serialize(&data).unwrap();let (admit,used)=budget(1<<20);
        let identity=decode_identity(&wire,Some(&admit)).unwrap();let clone=identity.clone();
        assert_eq!(bincode::serialize(&identity).unwrap(),wire);
        let standalone=bincode::serialize(&data.node).unwrap();
        let node=decode_node(&standalone,Some(&admit)).unwrap();
        let copied=copy_node(&node,Some(&admit)).unwrap();
        assert_eq!(bincode::serialize(&copied).unwrap(),standalone);
        drop(node);drop(copied);
        let held=used.load(SeqCst);drop(identity);drop(wire);drop(data);
        assert!(held>=4096);assert_eq!(used.load(SeqCst),held);assert_eq!(clone.node.attributes["zone"].len(),4096);
        drop(clone);assert_eq!(used.load(SeqCst),0);
        assert!(decode_identity(&u64::MAX.to_le_bytes(),Some(&admit)).is_err());
        assert_eq!(used.load(SeqCst),0);
    }
    #[test]
    fn error_depth_is_rejected_before_materialization_and_failure_keeps_no_charge() {
        let mut data=LegacyError{error_type:"e".into(),reason:"r".into(),status:500,metadata:Default::default(),caused_by:None};
        for _ in 0..128 { data=LegacyError{error_type:"e".into(),reason:"r".into(),status:500,metadata:Default::default(),caused_by:Some(Box::new(data))}; }
        let (admit,used)=budget(1<<20);
        assert!(decode_error(&bincode::serialize(&data).unwrap(),Some(&admit)).is_err());
        assert_eq!(used.load(SeqCst),0);
    }
}
