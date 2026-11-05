//! Shared recovery header text retains its admission through cloned targets.
use std::{sync::Arc,ops::Deref};
use serde::{Serialize,Deserialize};
use serde::de::{Visitor,SeqAccess,MapAccess,Error};
use elasticsearch_engine::{error::EsError,memory::{breakers,Reservation}};
use super::super::decode::Text;
#[derive(Debug)]
pub struct HeaderText {
pub index:String,
pub index_uuid:String,
pub settings_json:String,
pub mapping_json:String,
_memory:Reservation,
}
#[derive(Clone,Debug)]
pub struct RecoveryStartInfo {
pub shard:u32,
pub primary_term:u64,
pub start_seq_no:i64,
pub snapshot_max_seq_no:i64,
pub snapshot_docs:u64,
pub snapshot_bytes_estimate:u64,
pub translog_ops_on_start:u64,
text:Arc<HeaderText>,
}
impl Deref for RecoveryStartInfo {type Target=HeaderText;fn deref(&self)->&HeaderText{&self.text}}
impl PartialEq for RecoveryStartInfo {fn eq(&self,other:&Self)->bool{self.index==other.index && self.index_uuid==other.index_uuid && self.shard==other.shard && self.primary_term==other.primary_term && self.settings_json==other.settings_json && self.mapping_json==other.mapping_json && self.start_seq_no==other.start_seq_no && self.snapshot_max_seq_no==other.snapshot_max_seq_no && self.snapshot_docs==other.snapshot_docs && self.snapshot_bytes_estimate==other.snapshot_bytes_estimate && self.translog_ops_on_start==other.translog_ops_on_start}}
impl Eq for RecoveryStartInfo {}
impl RecoveryStartInfo {
#[allow(clippy::too_many_arguments)]
pub fn try_new(index:&str,index_uuid:&str,shard:u32,primary_term:u64,settings_json:&str,mapping_json:&str,start_seq_no:i64,snapshot_max_seq_no:i64,snapshot_docs:u64,snapshot_bytes_estimate:u64,translog_ops_on_start:u64)->Result<Self,EsError>{
let mut memory=breakers().reserve_request((std::mem::size_of::<HeaderText>()+2*std::mem::size_of::<usize>()) as u64,"recovery header owner")?;
memory.grow(index.len() as u64,"recovery header text")?;
memory.grow(index_uuid.len() as u64,"recovery header text")?;
memory.grow(settings_json.len() as u64,"recovery header text")?;
memory.grow(mapping_json.len() as u64,"recovery header text")?;
let index={let mut text=String::new();text.try_reserve_exact(index.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;text.push_str(index);text};
let index_uuid={let mut text=String::new();text.try_reserve_exact(index_uuid.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;text.push_str(index_uuid);text};
let settings_json={let mut text=String::new();text.try_reserve_exact(settings_json.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;text.push_str(settings_json);text};
let mapping_json={let mut text=String::new();text.try_reserve_exact(mapping_json.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;text.push_str(mapping_json);text};
Ok(Self{shard,primary_term,start_seq_no,snapshot_max_seq_no,snapshot_docs,snapshot_bytes_estimate,translog_ops_on_start,text:Arc::new(HeaderText{index,index_uuid,settings_json,mapping_json,_memory:memory.into_result_owner()})})
}
}
impl Serialize for RecoveryStartInfo {fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{use serde::ser::SerializeStruct;let mut out=s.serialize_struct("RecoveryStartInfo",11)?;out.serialize_field("index",&self.index)?;out.serialize_field("index_uuid",&self.index_uuid)?;out.serialize_field("shard",&self.shard)?;out.serialize_field("primary_term",&self.primary_term)?;out.serialize_field("settings_json",&self.settings_json)?;out.serialize_field("mapping_json",&self.mapping_json)?;out.serialize_field("start_seq_no",&self.start_seq_no)?;out.serialize_field("snapshot_max_seq_no",&self.snapshot_max_seq_no)?;out.serialize_field("snapshot_docs",&self.snapshot_docs)?;out.serialize_field("snapshot_bytes_estimate",&self.snapshot_bytes_estimate)?;out.serialize_field("translog_ops_on_start",&self.translog_ops_on_start)?;out.end()}}
impl<'de> Deserialize<'de> for RecoveryStartInfo {fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{struct V;impl<'de> Visitor<'de> for V {type Value=RecoveryStartInfo;fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("RecoveryStartInfo")}
fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{let mut memory=breakers().reserve_request((std::mem::size_of::<HeaderText>()+2*std::mem::size_of::<usize>()) as u64,"recovery header owner").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
let index_uuid=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index_uuid"))?;
let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
let settings_json=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("settings_json"))?;
let mapping_json=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("mapping_json"))?;
let start_seq_no=seq.next_element()?.ok_or_else(||A::Error::missing_field("start_seq_no"))?;
let snapshot_max_seq_no=seq.next_element()?.ok_or_else(||A::Error::missing_field("snapshot_max_seq_no"))?;
let snapshot_docs=seq.next_element()?.ok_or_else(||A::Error::missing_field("snapshot_docs"))?;
let snapshot_bytes_estimate=seq.next_element()?.ok_or_else(||A::Error::missing_field("snapshot_bytes_estimate"))?;
let translog_ops_on_start=seq.next_element()?.ok_or_else(||A::Error::missing_field("translog_ops_on_start"))?;
Ok(RecoveryStartInfo{shard,primary_term,start_seq_no,snapshot_max_seq_no,snapshot_docs,snapshot_bytes_estimate,translog_ops_on_start,text:Arc::new(HeaderText{index,index_uuid,settings_json,mapping_json,_memory:memory.into_result_owner()})})
}
fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{#[derive(Deserialize)]#[serde(field_identifier)]enum Field{#[serde(rename="index")]F0,#[serde(rename="index_uuid")]F1,#[serde(rename="shard")]F2,#[serde(rename="primary_term")]F3,#[serde(rename="settings_json")]F4,#[serde(rename="mapping_json")]F5,#[serde(rename="start_seq_no")]F6,#[serde(rename="snapshot_max_seq_no")]F7,#[serde(rename="snapshot_docs")]F8,#[serde(rename="snapshot_bytes_estimate")]F9,#[serde(rename="translog_ops_on_start")]F10,#[serde(other)]Ignore}
let mut memory=breakers().reserve_request((std::mem::size_of::<HeaderText>()+2*std::mem::size_of::<usize>()) as u64,"recovery header owner").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let mut index=None;
let mut index_uuid=None;
let mut shard=None;
let mut primary_term=None;
let mut settings_json=None;
let mut mapping_json=None;
let mut start_seq_no=None;
let mut snapshot_max_seq_no=None;
let mut snapshot_docs=None;
let mut snapshot_bytes_estimate=None;
let mut translog_ops_on_start=None;
while let Some(field)=map.next_key::<Field>()?{match field{Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));}index=Some(map.next_value_seed(Text(&mut memory))?);},Field::F1=>{if index_uuid.is_some(){return Err(A::Error::duplicate_field("index_uuid"));}index_uuid=Some(map.next_value_seed(Text(&mut memory))?);},Field::F2=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));}shard=Some(map.next_value()?);},Field::F3=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));}primary_term=Some(map.next_value()?);},Field::F4=>{if settings_json.is_some(){return Err(A::Error::duplicate_field("settings_json"));}settings_json=Some(map.next_value_seed(Text(&mut memory))?);},Field::F5=>{if mapping_json.is_some(){return Err(A::Error::duplicate_field("mapping_json"));}mapping_json=Some(map.next_value_seed(Text(&mut memory))?);},Field::F6=>{if start_seq_no.is_some(){return Err(A::Error::duplicate_field("start_seq_no"));}start_seq_no=Some(map.next_value()?);},Field::F7=>{if snapshot_max_seq_no.is_some(){return Err(A::Error::duplicate_field("snapshot_max_seq_no"));}snapshot_max_seq_no=Some(map.next_value()?);},Field::F8=>{if snapshot_docs.is_some(){return Err(A::Error::duplicate_field("snapshot_docs"));}snapshot_docs=Some(map.next_value()?);},Field::F9=>{if snapshot_bytes_estimate.is_some(){return Err(A::Error::duplicate_field("snapshot_bytes_estimate"));}snapshot_bytes_estimate=Some(map.next_value()?);},Field::F10=>{if translog_ops_on_start.is_some(){return Err(A::Error::duplicate_field("translog_ops_on_start"));}translog_ops_on_start=Some(map.next_value()?);},Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;}}}
let index=index.ok_or_else(||A::Error::missing_field("index"))?;
let index_uuid=index_uuid.ok_or_else(||A::Error::missing_field("index_uuid"))?;
let shard=shard.ok_or_else(||A::Error::missing_field("shard"))?;
let primary_term=primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?;
let settings_json=settings_json.ok_or_else(||A::Error::missing_field("settings_json"))?;
let mapping_json=mapping_json.ok_or_else(||A::Error::missing_field("mapping_json"))?;
let start_seq_no=start_seq_no.ok_or_else(||A::Error::missing_field("start_seq_no"))?;
let snapshot_max_seq_no=snapshot_max_seq_no.ok_or_else(||A::Error::missing_field("snapshot_max_seq_no"))?;
let snapshot_docs=snapshot_docs.ok_or_else(||A::Error::missing_field("snapshot_docs"))?;
let snapshot_bytes_estimate=snapshot_bytes_estimate.ok_or_else(||A::Error::missing_field("snapshot_bytes_estimate"))?;
let translog_ops_on_start=translog_ops_on_start.ok_or_else(||A::Error::missing_field("translog_ops_on_start"))?;
Ok(RecoveryStartInfo{shard,primary_term,start_seq_no,snapshot_max_seq_no,snapshot_docs,snapshot_bytes_estimate,translog_ops_on_start,text:Arc::new(HeaderText{index,index_uuid,settings_json,mapping_json,_memory:memory.into_result_owner()})})
}
}
d.deserialize_struct("RecoveryStartInfo",&["index","index_uuid","shard","primary_term","settings_json","mapping_json","start_seq_no","snapshot_max_seq_no","snapshot_docs","snapshot_bytes_estimate","translog_ops_on_start"],V)}}
#[cfg(test)]mod tests {
 use super::*;
 use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
 #[test]fn header_clone_shares_admitted_text_and_wire_order_is_unchanged(){
  // Private allocation tests exclude unrelated process RSS; child/parent limits remain finite.
  let budget=Breakers::leak_for_tests(1<<20);
  budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
  let scope=RequestScope::new("header",budget);
  let header=scoped(Some(scope.clone()),||RecoveryStartInfo::try_new("i","uuid",2,3,"{}",&"x".repeat(4096),4,5,6,7,8)).unwrap();
  let wire=bincode::serialize(&header).unwrap();
  let legacy=bincode::serialize(&("i","uuid",2u32,3u64,"{}","x".repeat(4096),4i64,5i64,6u64,7u64,8u64)).unwrap();
  assert_eq!(wire,legacy);
  let decoded:RecoveryStartInfo=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();assert_eq!(decoded,header);
  let copy=decoded.clone();assert_eq!(copy.mapping_json.as_ptr(),decoded.mapping_json.as_ptr());
  drop(header);drop(decoded);drop(scope);assert!(budget.get(Kind::Request).used()>=4096);
  assert_eq!(copy.mapping_json.len(),4096);drop(copy);assert_eq!(budget.get(Kind::Request).used(),0);
 }
 #[test]fn header_decode_refuses_before_large_text_copy_and_releases_prior_fields(){
  let wire=bincode::serialize(&("i","uuid",2u32,3u64,"{}","x".repeat(4096),4i64,5i64,6u64,7u64,8u64)).unwrap();
  let budget=Breakers::leak_for_tests(512);
  budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
  let scope=RequestScope::new("header denial",budget);
  assert!(scoped(Some(scope),||bincode::deserialize::<RecoveryStartInfo>(&wire)).is_err());
  assert_eq!(budget.get(Kind::Request).used(),0);
 }
}
