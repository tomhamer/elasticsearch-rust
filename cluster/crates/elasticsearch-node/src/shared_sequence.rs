//! Immutable typed wire arrays: slots are admitted before allocation, nested
//! values carry their own owners, and clones share rather than copy buffers.
use elasticsearch_engine::{error::EsError, memory::{breakers,Reservation}};
use serde::{Serialize,Serializer,Deserialize,Deserializer,de::{SeqAccess,Visitor,Error}};
use std::{sync::Arc,ops::Deref};
#[derive(Debug)]
struct Owned<T>{values:Vec<T>,_memory:Reservation}
#[derive(Debug)]
pub struct SharedSequence<T>(Option<Arc<Owned<T>>>);
impl<T> Clone for SharedSequence<T>{fn clone(&self)->Self{Self(self.0.clone())}}
impl<T> Default for SharedSequence<T>{fn default()->Self{Self(None)}}
impl<T> Deref for SharedSequence<T>{type Target=[T];fn deref(&self)->&[T]{self.0.as_ref().map_or(&[],|o|o.values.as_slice())}}
impl<T:PartialEq> PartialEq for SharedSequence<T>{fn eq(&self,other:&Self)->bool{**self==**other}}
impl<T:Serialize> Serialize for SharedSequence<T>{fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok,S::Error>{self.deref().serialize(s)}}
fn failure(e:impl std::fmt::Display)->EsError{EsError::new(429,"circuit_breaking_exception",e.to_string())}
pub(crate) struct Builder<T>{values:Vec<T>,memory:Reservation}
impl<T> Builder<T>{
 pub(crate) fn new()->Result<Self,EsError>{Ok(Self{values:Vec::new(),memory:breakers().reserve_request(0,"typed wire sequence")?})}
 pub(crate) fn push(&mut self,value:T)->Result<(),EsError>{
  if self.values.len()==self.values.capacity(){
   let cap=self.values.capacity().checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||failure("sequence capacity overflow"))?;
   let bytes=cap.checked_mul(std::mem::size_of::<T>()).ok_or_else(||failure("sequence bytes overflow"))?;
   // The old allocation remains charged during the replacement allocation.
   self.memory.grow(bytes as u64,"typed sequence replacement")?;
   let mut next=Vec::new();next.try_reserve_exact(cap).map_err(failure)?;
   next.append(&mut self.values);
   let old_bytes=self.values.capacity()*std::mem::size_of::<T>();
   let old=std::mem::replace(&mut self.values,next);drop(old);self.memory.shrink(old_bytes as u64);
  }
  self.values.push(value);Ok(())
 }
 pub(crate) fn finish(mut self)->Result<SharedSequence<T>,EsError>{
  self.memory.grow((std::mem::size_of::<Owned<T>>()+2*std::mem::size_of::<usize>())as u64,"typed sequence owner")?;
  Ok(SharedSequence(Some(Arc::new(Owned{values:self.values,_memory:self.memory.into_result_owner()}))))
 }
}
impl<T> SharedSequence<T>{
 pub fn try_generate(f:impl FnOnce(&mut dyn FnMut(T)->Result<(),EsError>)->Result<(),EsError>)->Result<Self,EsError>{let mut b=Builder::new()?;f(&mut |v|b.push(v))?;b.finish()}
 pub fn try_generate_sorted(f:impl FnOnce(&mut dyn FnMut(T)->Result<(),EsError>)->Result<(),EsError>)->Result<Self,EsError> where T:Ord {let mut b=Builder::new()?;f(&mut |v|b.push(v))?;b.values.sort_unstable();b.finish()}
 pub fn try_collect_results(values:impl IntoIterator<Item=Result<T,EsError>>)->Result<Self,EsError>{let mut b=Builder::new()?;for v in values{b.push(v?)?;}b.finish()}
 pub fn try_collect(values:impl IntoIterator<Item=T>)->Result<Self,EsError>{let mut b=Builder::new()?;for v in values{b.push(v)?;}b.finish()}
}
impl<'de,T:Deserialize<'de>> Deserialize<'de> for SharedSequence<T>{
 fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
  struct V<T>(std::marker::PhantomData<T>);
  impl<'de,T:Deserialize<'de>> Visitor<'de> for V<T>{
   type Value=SharedSequence<T>;
   fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("admitted wire sequence")}
   fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
    let err=|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e));
    let mut b=Builder::new().map_err(err)?;
    // Do not trust a peer's count or let Vec's serde preallocation bypass admission.
    while let Some(value)=seq.next_element()?{b.push(value).map_err(err)?;}
    b.finish().map_err(err)
   }
  }
  d.deserialize_seq(V(std::marker::PhantomData))
 }
}
pub struct IntoIter<T>{values:SharedSequence<T>,next:usize}
impl<T:Clone> Iterator for IntoIter<T>{type Item=T;fn next(&mut self)->Option<T>{let value=self.values.get(self.next)?.clone();self.next+=1;Some(value)}}
impl<T:Clone> IntoIterator for SharedSequence<T>{type Item=T;type IntoIter=IntoIter<T>;fn into_iter(self)->Self::IntoIter{IntoIter{values:self,next:0}}}
impl<'a,T> IntoIterator for &'a SharedSequence<T>{type Item=&'a T;type IntoIter=std::slice::Iter<'a,T>;fn into_iter(self)->Self::IntoIter{self.iter()}}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::front::SharedText;
 use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
 #[test]
 fn wire_sequence_preserves_layout_and_retains_shared_text_after_input_drop(){
  let budget=Breakers::leak_for_tests(1<<20);
  budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
  let scope=RequestScope::new("wire sequence test",budget);
  let raw=vec!["x".repeat(4096),"last".into()];let bytes=bincode::serialize(&raw).unwrap();
  let value:SharedSequence<SharedText>=scoped(Some(scope.clone()),||bincode::deserialize(&bytes)).unwrap();
  assert_eq!(bincode::serialize(&value).unwrap(),bytes);
  let clone=value.clone();assert_eq!(value.as_ptr(),clone.as_ptr());drop(value);drop(bytes);drop(raw);drop(scope);
  assert!(budget.get(Kind::Request).used()>=4100);assert_eq!(clone[0].len(),4096);drop(clone);assert_eq!(budget.get(Kind::Request).used(),0);
 }
 #[test]
 fn failed_nested_text_decode_releases_all_prior_array_and_text_owners(){
  let budget=Breakers::leak_for_tests(1024);
  budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
  let scope=RequestScope::new("wire refusal",budget);
  let bytes=bincode::serialize(&vec!["ok".into(),"x".repeat(4096)]).unwrap();
  let result:Result<SharedSequence<SharedText>,_>=scoped(Some(scope.clone()),||bincode::deserialize(&bytes));
  assert!(result.is_err());drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
 }
}

impl<T:PartialEq> PartialEq<Vec<T>> for SharedSequence<T>{fn eq(&self,other:&Vec<T>)->bool{&**self==other.as_slice()}}

impl<T> AsRef<[T]> for SharedSequence<T>{fn as_ref(&self)->&[T]{self}}
impl SharedSequence<u8> {
 /// Reserve the complete bounded output before the OS writes into it. Reads
 /// cannot grow the buffer, and truncation preserves capacity ownership.
 pub fn try_read_prefix(mut input:impl std::io::Read,limit:usize)->Result<Self,EsError>{
  let mut b=Builder::new()?;
  b.memory.grow(limit as u64,"snapshot chunk bytes")?;
  b.values.try_reserve_exact(limit).map_err(failure)?;
  b.values.resize(limit,0);let mut used=0;
  while used<limit {match input.read(&mut b.values[used..]){
   Ok(0)=>break,Ok(n)=>used+=n,Err(e) if e.kind()==std::io::ErrorKind::Interrupted=>continue,
   Err(e)=>return Err(EsError::new(500,"i_o_exception",e.to_string()))
  }}
  b.values.truncate(used);b.finish()
 }
}
