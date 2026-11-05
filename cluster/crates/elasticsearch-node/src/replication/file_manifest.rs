//! Shared manifest: typed strings and entry slots admitted before allocation.
use std::{ops::Deref,sync::Arc};
use serde::{Serialize,Deserialize};
use serde::de::{DeserializeSeed,Visitor,SeqAccess,MapAccess,Error};
use elasticsearch_engine::{hooks::{IndexManifest,files::IndexFileEntry},error::EsError,memory::{breakers,Reservation}};
use super::super::decode::Text;
#[derive(Debug)]struct Owned {value:IndexManifest,_memory:Reservation}
#[derive(Clone,Debug)]pub struct AdmittedManifest(Arc<Owned>);
impl Deref for AdmittedManifest {type Target=IndexManifest;fn deref(&self)->&IndexManifest{&self.0.value}}
impl Serialize for AdmittedManifest {fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{self.0.value.serialize(s)}}
impl AdmittedManifest {
 pub fn try_copy(v:&IndexManifest)->Result<Self,EsError>{
  let mut memory=breakers().reserve_request((std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>()) as u64,"recovery manifest owner")?;
  let slots=v.files.len().checked_mul(std::mem::size_of::<IndexFileEntry>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","manifest slots overflow"))?;
  memory.grow(slots as u64,"manifest entries")?;
  fn text(m:&mut Reservation,s:&str)->Result<String,EsError>{m.grow(s.len() as u64,"manifest text")?;let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;out.push_str(s);Ok(out)}
  let index=text(&mut memory,&v.index)?;let uuid=text(&mut memory,&v.uuid)?;
  let mut files=Vec::new();files.try_reserve_exact(v.files.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
  for f in &v.files {files.push(IndexFileEntry{path:text(&mut memory,&f.path)?,len:f.len,crc32:f.crc32});}
  Ok(Self(Arc::new(Owned{value:IndexManifest{index,uuid,files,committed_seq_no:v.committed_seq_no,local_checkpoint:v.local_checkpoint,max_seq_no:v.max_seq_no,primary_term:v.primary_term,includes_translog:v.includes_translog},_memory:memory.into_result_owner()})))
 }
}
struct Entry<'a>(&'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Entry<'_>{type Value=IndexFileEntry;fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Self::Value,D::Error>{
 struct V<'a>(&'a mut Reservation);impl<'de> Visitor<'de> for V<'_>{type Value=IndexFileEntry;fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("manifest entry")}
 fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{Ok(IndexFileEntry{path:seq.next_element_seed(Text(self.0))?.ok_or_else(||A::Error::missing_field("path"))?,len:seq.next_element()?.ok_or_else(||A::Error::missing_field("len"))?,crc32:seq.next_element()?.ok_or_else(||A::Error::missing_field("crc32"))?})}
 fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
 #[derive(Deserialize)]#[serde(field_identifier)]enum F{#[serde(rename="path")]Path,#[serde(rename="len")]Len,#[serde(rename="crc32")]Crc,#[serde(other)]Ignore}
 let(mut path,mut len,mut crc32)=(None,None,None);while let Some(f)=map.next_key::<F>()?{match f{F::Path=>{if path.is_some(){return Err(A::Error::duplicate_field("path"));}path=Some(map.next_value_seed(Text(self.0))?);},F::Len=>{if len.is_some(){return Err(A::Error::duplicate_field("len"));}len=Some(map.next_value()?);},F::Crc=>{if crc32.is_some(){return Err(A::Error::duplicate_field("crc32"));}crc32=Some(map.next_value()?);},F::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;}}}
 Ok(IndexFileEntry{path:path.ok_or_else(||A::Error::missing_field("path"))?,len:len.ok_or_else(||A::Error::missing_field("len"))?,crc32:crc32.ok_or_else(||A::Error::missing_field("crc32"))?})}
 }d.deserialize_struct("IndexFileEntry",&["path","len","crc32"],V(self.0))}}
struct Entries<'a>(&'a mut Reservation);
impl<'de> DeserializeSeed<'de> for Entries<'_>{type Value=Vec<IndexFileEntry>;fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Self::Value,D::Error>{
 struct V<'a>(&'a mut Reservation);impl<'de> Visitor<'de> for V<'_>{type Value=Vec<IndexFileEntry>;fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("manifest entries")}
 fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{let mut out=Vec::new();while let Some(entry)=seq.next_element_seed(Entry(self.0))?{
 if out.len()==out.capacity(){let old=out.capacity();let capacity=old.checked_mul(2).unwrap_or(usize::MAX).max(1);let bytes=capacity.checked_mul(std::mem::size_of::<IndexFileEntry>()).ok_or_else(||A::Error::custom("manifest size overflow"))?;
 self.0.grow(bytes as u64,"manifest entry replacement").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;out.try_reserve_exact(capacity-out.len()).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;self.0.shrink((old*std::mem::size_of::<IndexFileEntry>()) as u64);}
 out.push(entry);}Ok(out)}
 }d.deserialize_seq(V(self.0))}}
impl<'de> Deserialize<'de> for AdmittedManifest{fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{struct V;impl<'de> Visitor<'de> for V {type Value=AdmittedManifest;fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("IndexManifest")}
fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{let mut memory=breakers().reserve_request((std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>()) as u64,"manifest owner").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
let uuid=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("uuid"))?;
let files=seq.next_element_seed(Entries(&mut memory))?.ok_or_else(||A::Error::missing_field("files"))?;
let committed_seq_no=seq.next_element()?.ok_or_else(||A::Error::missing_field("committed_seq_no"))?;
let local_checkpoint=seq.next_element()?.ok_or_else(||A::Error::missing_field("local_checkpoint"))?;
let max_seq_no=seq.next_element()?.ok_or_else(||A::Error::missing_field("max_seq_no"))?;
let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
let includes_translog=seq.next_element()?.ok_or_else(||A::Error::missing_field("includes_translog"))?;
Ok(AdmittedManifest(Arc::new(Owned{value:IndexManifest{index,uuid,files,committed_seq_no,local_checkpoint,max_seq_no,primary_term,includes_translog},_memory:memory.into_result_owner()})))
}
fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{#[derive(Deserialize)]#[serde(field_identifier)]enum F{#[serde(rename="index")]F0,#[serde(rename="uuid")]F1,#[serde(rename="files")]F2,#[serde(rename="committed_seq_no")]F3,#[serde(rename="local_checkpoint")]F4,#[serde(rename="max_seq_no")]F5,#[serde(rename="primary_term")]F6,#[serde(rename="includes_translog")]F7,#[serde(other)]Ignore}
let mut memory=breakers().reserve_request((std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>()) as u64,"manifest owner").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let mut index=None;
let mut uuid=None;
let mut files=None;
let mut committed_seq_no=None;
let mut local_checkpoint=None;
let mut max_seq_no=None;
let mut primary_term=None;
let mut includes_translog=None;
while let Some(field)=map.next_key::<F>()?{match field{F::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));}index=Some(map.next_value_seed(Text(&mut memory))?);},F::F1=>{if uuid.is_some(){return Err(A::Error::duplicate_field("uuid"));}uuid=Some(map.next_value_seed(Text(&mut memory))?);},F::F2=>{if files.is_some(){return Err(A::Error::duplicate_field("files"));}files=Some(map.next_value_seed(Entries(&mut memory))?);},F::F3=>{if committed_seq_no.is_some(){return Err(A::Error::duplicate_field("committed_seq_no"));}committed_seq_no=Some(map.next_value()?);},F::F4=>{if local_checkpoint.is_some(){return Err(A::Error::duplicate_field("local_checkpoint"));}local_checkpoint=Some(map.next_value()?);},F::F5=>{if max_seq_no.is_some(){return Err(A::Error::duplicate_field("max_seq_no"));}max_seq_no=Some(map.next_value()?);},F::F6=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));}primary_term=Some(map.next_value()?);},F::F7=>{if includes_translog.is_some(){return Err(A::Error::duplicate_field("includes_translog"));}includes_translog=Some(map.next_value()?);},F::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;}}}
let index=index.ok_or_else(||A::Error::missing_field("index"))?;
let uuid=uuid.ok_or_else(||A::Error::missing_field("uuid"))?;
let files=files.ok_or_else(||A::Error::missing_field("files"))?;
let committed_seq_no=committed_seq_no.ok_or_else(||A::Error::missing_field("committed_seq_no"))?;
let local_checkpoint=local_checkpoint.ok_or_else(||A::Error::missing_field("local_checkpoint"))?;
let max_seq_no=max_seq_no.ok_or_else(||A::Error::missing_field("max_seq_no"))?;
let primary_term=primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?;
let includes_translog=includes_translog.ok_or_else(||A::Error::missing_field("includes_translog"))?;
Ok(AdmittedManifest(Arc::new(Owned{value:IndexManifest{index,uuid,files,committed_seq_no,local_checkpoint,max_seq_no,primary_term,includes_translog},_memory:memory.into_result_owner()})))
}
}
d.deserialize_struct("IndexManifest",&["index","uuid","files","committed_seq_no","local_checkpoint","max_seq_no","primary_term","includes_translog"],V)}}
#[cfg(test)]mod tests{
 use super::*;use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
 fn original()->IndexManifest{IndexManifest{index:"index".into(),uuid:"uuid".into(),files:vec![IndexFileEntry{path:"x".repeat(4096),len:3,crc32:7}],committed_seq_no:1,local_checkpoint:2,max_seq_no:3,primary_term:4,includes_translog:false}}
 #[test]fn manifest_wire_and_shared_lifetime(){
  // Private allocation tests exclude unrelated process RSS; child/parent limits remain finite.
  let old=original();let wire=bincode::serialize(&old).unwrap();let budget=Breakers::leak_for_tests(1<<20);
  budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
  let scope=RequestScope::new("manifest",budget);
  let value:AdmittedManifest=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();assert_eq!(bincode::serialize(&value).unwrap(),wire);
  let copied=scoped(Some(scope.clone()),||AdmittedManifest::try_copy(&old)).unwrap();assert_eq!(&*copied,&old);
  let shared=value.clone();assert_eq!(shared.files.as_ptr(),value.files.as_ptr());drop(value);drop(copied);drop(scope);drop(wire);drop(old);
  assert!(budget.get(Kind::Request).used()>=4096);assert_eq!(shared.files[0].len,3);drop(shared);assert_eq!(budget.get(Kind::Request).used(),0);
 }
 #[test]fn manifest_large_path_refuses_and_unwinds_prior_entries(){
  let mut old=original();old.files.insert(0,IndexFileEntry{path:"small".into(),len:0,crc32:0});let wire=bincode::serialize(&old).unwrap();let budget=Breakers::leak_for_tests(512);
  budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
  let scope=RequestScope::new("manifest refusal",budget);
  assert!(scoped(Some(scope),||bincode::deserialize::<AdmittedManifest>(&wire)).is_err());assert_eq!(budget.get(Kind::Request).used(),0);
 }
}
