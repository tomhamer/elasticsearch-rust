//! Owner-carrying projection of persisted CopyRecord for store discovery. The
//! wire layout stays identical, while cache clones share actual text/array owners.
use crate::{front::SharedText, shared_sequence::{Builder,SharedSequence}};
use elasticsearch_engine::{error::{EsError,Result},memory::breakers};
use serde::{Serialize,Deserialize};
use std::{path::Path,io::Read};
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct StoreRecord {
 pub index:SharedText,pub index_uuid:SharedText,pub shard:u32,pub allocation_id:SharedText,
 pub primary_term:u64,pub primary:bool,
 #[serde(default)] pub history_uuid:Option<SharedText>,
 #[serde(default)] pub global_checkpoint:Option<i64>,
}
#[derive(Clone,Debug,Serialize)]
#[serde(transparent)]
pub struct StoreRecords(SharedSequence<StoreRecord>);
impl std::ops::Deref for StoreRecords{type Target=[StoreRecord];fn deref(&self)->&Self::Target{&self.0}}
impl<'a> IntoIterator for &'a StoreRecords{type Item=&'a StoreRecord;type IntoIter=std::slice::Iter<'a,StoreRecord>;fn into_iter(self)->Self::IntoIter{self.iter()}}
impl<'de> Deserialize<'de> for StoreRecords{fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{crate::control_memory::decode(d).map(Self)}}
fn failure(message:impl Into<String>)->EsError{EsError::new(429,"circuit_breaking_exception",message)}
pub(crate) fn read(path:&Path)->Result<Option<StoreRecord>>{elasticsearch_engine::memory::native_control(||read_native(path))}
fn read_native(path:&Path)->Result<Option<StoreRecord>>{
 let path_bytes=path.as_os_str().len().checked_add(32).ok_or_else(||failure("store record path overflow"))?;
 let _path=breakers().reserve_native(path_bytes as u64,"store record path")?;
 let mut storage=std::ffi::OsString::new();storage.try_reserve_exact(path_bytes).map_err(|e|failure(e.to_string()))?;storage.push(path.as_os_str());
 let mut filename=std::path::PathBuf::from(storage);filename.push("cluster-copy.json");
 let file=match std::fs::File::open(&filename){Ok(file)=>file,Err(_)=>return Ok(None)};
 let length=usize::try_from(file.metadata().map_err(|e|EsError::internal(e.to_string()))?.len()).map_err(|_|failure("store record size overflow"))?;
 // One extra byte detects a growing file without read_to_end reallocating.
 let capacity=length.checked_add(1).ok_or_else(||failure("store record size overflow"))?;
 let _wire=breakers().reserve_native(capacity as u64,"store record read")?;
 let mut bytes=Vec::new();bytes.try_reserve_exact(capacity).map_err(|e|failure(e.to_string()))?;bytes.resize(capacity,0);
 let mut file=file.take(capacity as u64);let mut used=0;
 while used<capacity{let n=file.read(&mut bytes[used..]).map_err(|e|EsError::internal(e.to_string()))?;if n==0{break}used+=n;}
 if used>length{return Err(EsError::new(503,"unavailable_shards_exception","store record changed while reading"));}
 // serde_json may geometrically grow its escape scratch before the seeded
 // SharedText visitor is called. Cover old/new scratch overlap separately from
 // both file bytes and the retained decoded fields (same bound as OwnedJson).
 let scratch=(used as u64).checked_mul(4).and_then(|n|n.checked_add(256)).ok_or_else(||failure("store JSON scratch overflow"))?;
 let _scratch=breakers().reserve_native(scratch,"store JSON escape scratch")?;
 // Invalid/missing durable records retain the existing unavailable-copy rule;
 // allocation denial must be surfaced, never silently omitted as a missing copy.
 match serde_json::from_slice(&bytes[..used]) {Ok(value)=>Ok(Some(value)),Err(error)=>{
  if error.to_string().starts_with("rustic.internal.decode-admission.v1:") {Err(failure(error.to_string()))} else {Ok(None)}
 }}
}
pub(crate) fn collect(node:&crate::node::ClusterNode)->Result<StoreRecords>{elasticsearch_engine::memory::native_control(||collect_native(node))}
fn collect_native(node:&crate::node::ClusterNode)->Result<StoreRecords>{
 let mut out=Builder::new()?;
 node.try_visit_shard_engines(|_,engine|engine.try_visit_indices(|svc|{if let Some(record)=read(svc.path())?{out.push(record)?;}Ok::<_,EsError>(())}))?;
 out.finish().map(StoreRecords)
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn records_keep_original_disk_and_wire_fields(){
  let original=crate::shards::CopyRecord{index:"i".into(),index_uuid:"uuid".into(),shard:2,allocation_id:"aid".into(),primary_term:7,primary:true,history_uuid:Some("hist".into()),global_checkpoint:Some(10)};
  let wire=bincode::serialize(&vec![original.clone()]).unwrap();let admitted:StoreRecords=bincode::deserialize(&wire).unwrap();
  assert_eq!(bincode::serialize(&admitted).unwrap(),wire);assert_eq!(&*admitted[0].index,"i");
  let disk:StoreRecord=serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();assert_eq!(disk.global_checkpoint,Some(10));
 }
}

/// The master cache owns its map/key allocations independently of fetch futures.
#[derive(Default)]
pub struct StoreCache { values:std::collections::BTreeMap<String,StoreRecords>, memory:Option<elasticsearch_engine::memory::Reservation> }
impl std::ops::Deref for StoreCache{type Target=std::collections::BTreeMap<String,StoreRecords>;fn deref(&self)->&Self::Target{&self.values}}
impl StoreCache {
 pub(crate) fn try_insert(&mut self,id:&str,records:StoreRecords)->Result<()> {
  let bytes=id.len().checked_add(2*(11*std::mem::size_of::<(String,StoreRecords)>()+20*std::mem::size_of::<usize>())).ok_or_else(||failure("store cache overflow"))?;
  if self.memory.is_none(){self.memory=Some(breakers().reserve_native(0,"store cache")?.into_result_owner());}
  self.memory.as_mut().unwrap().grow(bytes as u64,"store cache node and key")?;
  let key=crate::mapping_update::copy_wire_text(id)?;self.values.insert(key,records);Ok(())
 }
}
