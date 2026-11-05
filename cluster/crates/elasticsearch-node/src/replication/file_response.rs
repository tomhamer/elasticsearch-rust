//! File recovery response scalar admission; nested owners remain independent.
use serde::{Serialize,Deserialize};use serde::de::{Visitor,SeqAccess,MapAccess,Error};
use elasticsearch_engine::{error::EsError,memory::{breakers,Reservation}};
use super::super::decode::{Text,OptionalText};
#[derive(Debug,Serialize)]pub struct FileRecoveryStartResp {
pub recovery_id:u64,
pub info:super::super::recovery::RecoveryStartInfo,
pub manifest:super::AdmittedManifest,
pub history_uuid:Option<String>,
pub source_allocation_id:String,
pub primary_term:u64,
#[serde(skip)]_memory:Reservation,
}
impl FileRecoveryStartResp {pub fn try_new(recovery_id:u64,info:super::super::recovery::RecoveryStartInfo,manifest:super::AdmittedManifest,history_uuid:Option<&str>,source_allocation_id:&str,primary_term:u64)->Result<Self,EsError>{
let mut memory=breakers().reserve_request(0,"file recovery response text")?;
fn text(memory:&mut Reservation,value:&str)->Result<String,EsError>{memory.grow(value.len() as u64,"file response text")?;let mut out=String::new();out.try_reserve_exact(value.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;out.push_str(value);Ok(out)}
let history_uuid=history_uuid.map(|v|text(&mut memory,v)).transpose()?;let source_allocation_id=text(&mut memory,source_allocation_id)?;
Ok(Self{recovery_id,info,manifest,history_uuid,source_allocation_id,primary_term,_memory:memory.into_result_owner()})}}
impl<'de> Deserialize<'de> for FileRecoveryStartResp{fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{struct V;impl<'de> Visitor<'de> for V {type Value=FileRecoveryStartResp;fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("FileRecoveryStartResp")}
fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{let mut memory=breakers().reserve_request(0,"file recovery response text").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let recovery_id=seq.next_element()?.ok_or_else(||A::Error::missing_field("recovery_id"))?;
let info=seq.next_element()?.ok_or_else(||A::Error::missing_field("info"))?;
let manifest=seq.next_element()?.ok_or_else(||A::Error::missing_field("manifest"))?;
let history_uuid=seq.next_element_seed(OptionalText(&mut memory))?.ok_or_else(||A::Error::missing_field("history_uuid"))?;
let source_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("source_allocation_id"))?;
let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
Ok(FileRecoveryStartResp{recovery_id,info,manifest,history_uuid,source_allocation_id,primary_term,_memory:memory.into_result_owner()})
}
fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{#[derive(Deserialize)]#[serde(field_identifier)]enum F{#[serde(rename="recovery_id")]F0,#[serde(rename="info")]F1,#[serde(rename="manifest")]F2,#[serde(rename="history_uuid")]F3,#[serde(rename="source_allocation_id")]F4,#[serde(rename="primary_term")]F5,#[serde(other)]Ignore}
let mut memory=breakers().reserve_request(0,"file recovery response text").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let mut recovery_id=None;
let mut info=None;
let mut manifest=None;
let mut history_uuid=None;
let mut source_allocation_id=None;
let mut primary_term=None;
while let Some(field)=map.next_key::<F>()?{match field{F::F0=>{if recovery_id.is_some(){return Err(A::Error::duplicate_field("recovery_id"));}recovery_id=Some(map.next_value()?);},F::F1=>{if info.is_some(){return Err(A::Error::duplicate_field("info"));}info=Some(map.next_value()?);},F::F2=>{if manifest.is_some(){return Err(A::Error::duplicate_field("manifest"));}manifest=Some(map.next_value()?);},F::F3=>{if history_uuid.is_some(){return Err(A::Error::duplicate_field("history_uuid"));}history_uuid=Some(map.next_value_seed(OptionalText(&mut memory))?);},F::F4=>{if source_allocation_id.is_some(){return Err(A::Error::duplicate_field("source_allocation_id"));}source_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},F::F5=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));}primary_term=Some(map.next_value()?);},F::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;}}}
let recovery_id=recovery_id.ok_or_else(||A::Error::missing_field("recovery_id"))?;
let info=info.ok_or_else(||A::Error::missing_field("info"))?;
let manifest=manifest.ok_or_else(||A::Error::missing_field("manifest"))?;
let history_uuid=history_uuid.unwrap_or(None);
let source_allocation_id=source_allocation_id.ok_or_else(||A::Error::missing_field("source_allocation_id"))?;
let primary_term=primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?;
Ok(FileRecoveryStartResp{recovery_id,info,manifest,history_uuid,source_allocation_id,primary_term,_memory:memory.into_result_owner()})
}
}
d.deserialize_struct("FileRecoveryStartResp",&["recovery_id","info","manifest","history_uuid","source_allocation_id","primary_term"],V)}}
#[cfg(test)]mod tests{
 use super::*;use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
 #[test]fn nested_response_wire_and_last_owner_lifetime(){
  // Private allocation tests exclude unrelated process RSS; child/parent limits remain finite.
  let budget=Breakers::leak_for_tests(1<<20);
  budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
  let scope=RequestScope::new("file response",budget);
  let response=scoped(Some(scope.clone()),||{
   let info=super::super::super::recovery::RecoveryStartInfo::try_new("i","u",0,1,"{}","{}",1,0,0,0,0).unwrap();
   let manifest=super::super::AdmittedManifest::try_copy(&elasticsearch_engine::hooks::IndexManifest{index:"i".into(),uuid:"u".into(),files:vec![],committed_seq_no:0,local_checkpoint:0,max_seq_no:0,primary_term:1,includes_translog:false}).unwrap();
   FileRecoveryStartResp::try_new(7,info,manifest,Some("history"),"source",1).unwrap()
  });
  let bytes=bincode::serialize(&response).unwrap();let old=bincode::serialize(&(7u64,&response.info,&*response.manifest,Some("history"),"source",1u64)).unwrap();assert_eq!(bytes,old);
  let decoded:FileRecoveryStartResp=scoped(Some(scope.clone()),||bincode::deserialize(&bytes)).unwrap();
  let manifest=decoded.manifest.clone();let info=decoded.info.clone();drop(response);drop(decoded);drop(scope);
  assert_eq!(info.index,"i");assert_eq!(manifest.uuid,"u");assert!(budget.get(Kind::Request).used()>0);drop(info);drop(manifest);assert_eq!(budget.get(Kind::Request).used(),0);
 }
}
