//! Owned file-chunk bytes: reserve before decoding or reading the source.
use elasticsearch_engine::{error::EsError,memory::{breakers,Reservation}};
use serde::{Serialize,Deserialize};
use serde::de::{Visitor,SeqAccess,Error};
#[derive(Debug)]
pub struct ChunkBytes {value:Vec<u8>, _memory:Reservation, _source_memory:Option<elasticsearch_engine::rustic_core::admission::Reservation>}
impl std::ops::Deref for ChunkBytes {type Target=[u8];fn deref(&self)->&[u8]{&self.value}}
impl PartialEq for ChunkBytes {fn eq(&self,other:&Self)->bool{self.value==other.value}}
impl Eq for ChunkBytes {}
impl ChunkBytes {
    pub(crate) fn read_owned(memory:Reservation, read:impl FnOnce()->Result<Vec<u8>,String>)->Result<Self,String>{
        let bytes=read()?;Ok(Self::from_reserved(bytes,memory))
    }
    pub(crate) fn from_admitted(bytes:elasticsearch_engine::rustic_core::admission::Allocation<Vec<u8>>,memory:Reservation)->Self {
        let (value,source)=bytes.into_parts();Self{value,_memory:memory.into_result_owner(),_source_memory:Some(source)}
    }
    pub fn as_mut_slice(&mut self)->&mut[u8]{&mut self.value}
    pub fn reserve(bytes:u64)->Result<Reservation,EsError>{breakers().reserve_request(bytes,"file recovery chunk")}
    pub fn from_reserved(bytes:Vec<u8>,memory:Reservation)->Self {
        assert!(memory.bytes()>=bytes.capacity() as u64,"file chunk must be admitted before allocation");
        Self{value:bytes,_memory:memory.into_result_owner(),_source_memory:None}
    }
    pub fn copy_from(bytes:&[u8])->Result<Self,EsError>{
        let memory=Self::reserve(bytes.len() as u64)?;
        let mut out=Vec::new();out.try_reserve_exact(bytes.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        out.extend_from_slice(bytes);Ok(Self::from_reserved(out,memory))
    }
}
impl Serialize for ChunkBytes {fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{s.serialize_bytes(&self.value)}}
impl<'de> Deserialize<'de> for ChunkBytes {
 fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
  struct V;
  impl<'de> Visitor<'de> for V {
   type Value=ChunkBytes;
   fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("admitted file chunk bytes")}
   fn visit_bytes<E:Error>(self,v:&[u8])->Result<ChunkBytes,E>{ChunkBytes::copy_from(v).map_err(|e|E::custom(elasticsearch_transport::decode_admission_error(e)))}
   fn visit_borrowed_bytes<E:Error>(self,v:&'de[u8])->Result<ChunkBytes,E>{self.visit_bytes(v)}
   fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<ChunkBytes,A::Error>{
    let mut memory=ChunkBytes::reserve(0).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
    let mut out=Vec::new();
    while let Some(byte)=seq.next_element::<u8>()? {
     if out.len()==out.capacity(){
      let old=out.capacity();let capacity=old.checked_mul(2).unwrap_or(usize::MAX).max(1);
      memory.grow(capacity as u64,"file chunk replacement").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
      out.try_reserve_exact(capacity-out.len()).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
      memory.shrink(old as u64);
     }
     out.push(byte);
    }
    Ok(ChunkBytes::from_reserved(out,memory))
   }
  }
  d.deserialize_bytes(V)
 }
}
#[cfg(test)]
mod tests {
 use super::*;
 use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
 fn private_budget(bytes:u64)->&'static Breakers {
  let budget=Breakers::leak_for_tests(bytes);
  // These fixtures measure chunk ownership/refusal, not unrelated process RSS.
  budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
  budget
 }
 #[tokio::test]
 async fn canceled_waiter_keeps_chunk_admission_with_actual_reader_worker(){
  let budget=private_budget(8192);let scope=RequestScope::new("chunk canceled",budget);
  let memory=scoped(Some(scope.clone()),||ChunkBytes::reserve(4096)).unwrap().into_result_owner();
  let (started,ready)=tokio::sync::oneshot::channel();
  let (release,wait)=std::sync::mpsc::channel::<()>();
  let (finished,done)=tokio::sync::oneshot::channel();
  let task=tokio::spawn(async move {
   elasticsearch_http::pools::run(elasticsearch_http::pools::Pool::Recovery,move||{
    let chunk=ChunkBytes::read_owned(memory,||{
     let _=started.send(());let _=wait.recv();Ok(vec![0u8;4096])
    });
    drop(chunk);let _=finished.send(());
   }).await
  });
  ready.await.unwrap();task.abort();let _=task.await;drop(scope);
  assert_eq!(budget.get(Kind::Request).used(),4096,"worker still owns pre-read charge after cancellation");
  drop(release);done.await.unwrap();assert_eq!(budget.get(Kind::Request).used(),0);
 }
 #[test]
 fn chunk_wire_is_unchanged_and_owned_bytes_survive_input_and_scope() {
  let original=vec![37u8;4096];let wire=bincode::serialize(&original).unwrap();
  let budget=private_budget(1<<20);let scope=RequestScope::new("chunk",budget);
  let chunk:ChunkBytes=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();
  assert_eq!(bincode::serialize(&chunk).unwrap(),wire);drop(wire);drop(original);drop(scope);
  assert_eq!(chunk.len(),4096);assert!(budget.get(Kind::Request).used()>=4096);
  drop(chunk);assert_eq!(budget.get(Kind::Request).used(),0);
 }
 #[test]
 fn chunk_rejection_releases_partial_json_growth_and_binary_refuses_before_copy() {
  let budget=private_budget(128);let scope=RequestScope::new("chunk refusal",budget);
  let wire=bincode::serialize(&vec![0u8;4096]).unwrap();
  assert!(scoped(Some(scope.clone()),||bincode::deserialize::<ChunkBytes>(&wire)).is_err());
  assert_eq!(budget.get(Kind::Request).used(),0);
  let json=serde_json::to_string(&vec![0u8;4096]).unwrap();
  assert!(scoped(Some(scope),||serde_json::from_str::<ChunkBytes>(&json)).is_err());
  assert_eq!(budget.get(Kind::Request).used(),0);
 }
}
