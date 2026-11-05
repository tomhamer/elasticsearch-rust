//! Admission for the two concrete owned telemetry schemas decoded from JSON.
//! JSON is first decoded by the seeded owner; the typed copy is independently
//! charged before serde creates its strings/maps and retains that charge.
use elasticsearch_engine::{error::EsError,memory::{self,Reservation}};
use serde::de::DeserializeOwned;
use serde_json::Value;
pub(crate) struct Typed<T>{value:T,memory:Reservation}
impl<T> std::ops::Deref for Typed<T>{type Target=T;fn deref(&self)->&T{&self.value}}
fn overflow()->EsError{EsError::new(429,"circuit_breaking_exception","telemetry allocation overflow")}
fn bound<T>(v:&Value)->Result<u64,EsError>{
 // T is CcsStats or NodeGpuStats. Every nested record is no larger than T;
 // the two-node-per-entry bound covers BTree leaf/internal split storage.
 let unit=2usize.checked_mul(11usize.checked_mul(std::mem::size_of::<T>()+std::mem::size_of::<String>()).and_then(|n|n.checked_add(20*std::mem::size_of::<usize>())).ok_or_else(overflow)?).ok_or_else(overflow)?;
 fn walk(v:&Value,unit:usize)->Result<usize,EsError>{
  match v {
   Value::String(s)=>Ok(s.len()),
   Value::Array(a)=>a.iter().try_fold(a.len().checked_mul(unit).ok_or_else(overflow)?,|n,v|n.checked_add(walk(v,unit)?).ok_or_else(overflow)),
   Value::Object(m)=>m.iter().try_fold(m.len().checked_mul(unit).ok_or_else(overflow)?,|n,(k,v)|n.checked_add(k.len()).and_then(|n|n.checked_add(walk(v,unit).ok()?)).ok_or_else(overflow)),
   _=>Ok(0),
  }
 }
 u64::try_from(walk(v,unit)?).map_err(|_|overflow())
}
impl<T:DeserializeOwned+serde::Serialize> Typed<T>{
 pub(crate) fn parse(text:&str)->Result<Self,EsError>{
  let parsed=crate::owned_json::OwnedJson::parse(text)?;
  let memory=memory::breakers().reserve_request(bound::<T>(&parsed)?,"typed telemetry JSON objects")?;
  let _scratch=memory::breakers().reserve_request((text.len() as u64).checked_mul(4).and_then(|n|n.checked_add(256)).ok_or_else(overflow)?,"typed telemetry parser scratch")?;
  let value=serde_json::from_str(text).map_err(|e|EsError::new(500,"transport_serialization_exception",e.to_string()))?;
  Ok(Self{value,memory:memory.into_result_owner()})
 }
 pub(crate) fn copy(value:&T)->Result<Self,EsError>{let text=crate::front::SharedText::try_json(value)?;Self::parse(&text)}
 pub(crate) fn merge(&mut self,other:&Self,merge:impl FnOnce(&mut T,&T))->Result<(),EsError>{
  let text=crate::front::SharedText::try_json(&other.value)?;
  let parsed=crate::owned_json::OwnedJson::parse(&text)?;
  self.memory.grow(bound::<T>(&parsed)?,"telemetry merged object growth")?;
  merge(&mut self.value,&other.value);Ok(())
 }
}

/// Build a fresh bounded tree rather than cloning hidden map spare capacity.
pub(crate) fn copy_json(value:&Value,memory:&mut Reservation)->Result<Value,EsError>{
 fn text(s:&str,m:&mut Reservation)->Result<String,EsError>{m.grow(s.len() as u64,"telemetry text")?;let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;out.push_str(s);Ok(out)}
 Ok(match value {
  Value::String(s)=>Value::String(text(s,memory)?),
  Value::Array(a)=>{
   memory.grow(a.len().checked_mul(std::mem::size_of::<Value>()).ok_or_else(overflow)? as u64,"telemetry array")?;
   let mut out=Vec::new();out.try_reserve_exact(a.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
   for v in a {out.push(copy_json(v,memory)?)}Value::Array(out)
  },
  Value::Object(o)=>{
   let mut out=serde_json::Map::new();
   for(k,v)in o {memory.grow((2*(12*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>()))as u64,"telemetry object node")?;out.insert(text(k,memory)?,copy_json(v,memory)?);}
   Value::Object(out)
  },
  v=>v.clone(),
 })
}
pub(crate) fn owned_json(value:&Value)->Result<crate::owned_json::OwnedJson,EsError>{
 let mut memory=memory::breakers().reserve_request(0,"telemetry JSON")?;
 let value=copy_json(value,&mut memory)?;
 Ok(crate::owned_json::OwnedJson::from_reserved(value,memory))
}
impl<T:DeserializeOwned+serde::Serialize> Typed<T>{
 pub(crate) fn render(&self,render:impl FnOnce(&T)->Value)->Result<crate::owned_json::OwnedJson,EsError>{
  let text=crate::front::SharedText::try_json(&self.value)?;
  let parsed=crate::owned_json::OwnedJson::parse(&text)?;
  // Concrete CCS/GPU renderers copy map keys and scalar records; they create
  // at most 64 fixed root/summary fields in addition to the input tree.
  let fixed=64*(2*(12*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>())+128);
  let bytes=bound::<T>(&parsed)?.checked_mul(2).and_then(|n|n.checked_add(fixed as u64)).ok_or_else(overflow)?;
  let memory=memory::breakers().reserve_request(bytes,"telemetry rendered projection")?;
  let value=render(&self.value);
  Ok(crate::owned_json::OwnedJson::from_reserved(value,memory))
 }
}
/// Telemetry reducers only add fields/entries (never delete and retain hidden
/// spare capacity). Cover their old/new container overlap and copied strings
/// from both existing and incoming trees before invoking those reducers.
pub(crate) fn plan_merge(target:&Value,source:&Value,memory:&mut Reservation)->Result<(),EsError>{
 let bytes=bound::<Value>(target)?.checked_add(bound::<Value>(source)?).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(4096)).ok_or_else(overflow)?;
 memory.grow(bytes,"telemetry monotonic reducer workspace")
}
impl<T> Typed<T>{pub(crate) fn from_reserved(value:T,memory:Reservation)->Self{Self{value,memory:memory.into_result_owner()}}}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn ccs_typed_decode_preserves_u128_wire_sum(){
  let stats=elasticsearch_engine::ccs_stats::CcsStats::default();
  let wire=serde_json::to_string(&stats).unwrap().replace("\"sum\":0","\"sum\":18446744073709551617");
  let decoded=Typed::<elasticsearch_engine::ccs_stats::CcsStats>::parse(&wire).unwrap();
  assert_eq!(serde_json::to_string(&*decoded).unwrap(),wire);
 }
 #[test]
 fn typed_gpu_strings_survive_wire_drop(){
  let wire=String::from(r#"{"gpu_supported":true,"gpu_setting_enabled":true,"gpu_usage_count":3,"total_gpu_memory_in_bytes":99,"gpu_name":"device-λ"}"#);
  let decoded=Typed::<elasticsearch_engine::gpu::NodeGpuStats>::parse(&wire).unwrap();drop(wire);
  assert_eq!(decoded.gpu_name.as_deref(),Some("device-λ"));assert_eq!(decoded.gpu_usage_count,3);
 }
}
