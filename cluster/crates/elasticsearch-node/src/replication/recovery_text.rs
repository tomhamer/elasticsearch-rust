//! Immutable recovery telemetry text: snapshots share the actual allocation.
//! Completed recovery slots outlive requests. Charge their real backing to the
//! finite native/control budget, not the active request-work gauge. This keeps
//! aggregate admission/provenance and final-owner release intact.
use elasticsearch_engine::{error::EsError,memory::{breakers,Reservation}};
use serde::{Serialize,Deserialize,de::{DeserializeSeed,Error}};
use std::{sync::Arc,ops::Deref};
struct Owned { value:String, _memory:Reservation }
#[derive(Clone)]
pub struct RecoveryText(Arc<Owned>);
impl RecoveryText {
    pub fn try_new(value:&str)->Result<Self,EsError>{
        let mut memory=breakers().reserve_native((std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>()) as u64,"recovery telemetry text")?;
        memory.grow(value.len() as u64,"recovery telemetry text")?;
        let mut text=String::new();text.try_reserve_exact(value.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;text.push_str(value);
        Ok(Self(Arc::new(Owned{value:text,_memory:memory.into_result_owner()})))
    }
}
impl AsRef<str> for RecoveryText {fn as_ref(&self)->&str{&self.0.value}}
impl std::fmt::Display for RecoveryText {fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(&self.0.value)}}
impl Deref for RecoveryText {type Target=str;fn deref(&self)->&str{&self.0.value}}
impl std::fmt::Debug for RecoveryText {fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{self.0.value.fmt(f)}}
impl PartialEq for RecoveryText {fn eq(&self,other:&Self)->bool{self.0.value==other.0.value}}
impl Eq for RecoveryText {}
impl Serialize for RecoveryText {fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{s.serialize_str(self)}}
impl<'de> Deserialize<'de> for RecoveryText {fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
    let mut memory=breakers().reserve_native((std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>()) as u64,"recovery telemetry text").map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
    let value=super::super::decode::Text(&mut memory).deserialize(d)?;
    Ok(Self(Arc::new(Owned{value,_memory:memory.into_result_owner()})))
}}
#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::{self,Breakers,Kind,RequestScope};
    #[test]
    fn telemetry_clones_share_text_owner_and_preserve_string_wire() {
        let budget=Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("telemetry",budget);
        let text=memory::scoped(Some(scope.clone()),||RecoveryText::try_new("node-id")).unwrap();
        let copy=text.clone();assert!(Arc::ptr_eq(&text.0,&copy.0));
        let wire=bincode::serialize(&text).unwrap();assert_eq!(wire,bincode::serialize("node-id").unwrap());
        let decoded:RecoveryText=memory::scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();assert_eq!(text,decoded);
        drop(text);drop(decoded);drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
        assert!(budget.get(Kind::Accounting).used()>0);
        drop(copy);assert_eq!(budget.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn completed_recovery_text_retains_exact_native_charge_after_request_scope() {
        let budget=Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("completed recovery",budget);
        let texts=memory::scoped(Some(scope.clone()),|| {
            ["brk","DHCSHshrTsCXCYSz_TGdKw","V7-xp61-QMan7dzde8fdJw"]
                .map(|text|RecoveryText::try_new(text).unwrap())
        });
        let retained=texts.clone();
        let expected=3*(std::mem::size_of::<Owned>()+2*std::mem::size_of::<usize>())+3+22+22;
        drop(texts);drop(scope);
        assert_eq!(budget.get(Kind::Request).used(),0);
        assert_eq!(budget.get(Kind::Accounting).used(),expected as u64);
        drop(retained);
        assert_eq!(budget.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn telemetry_refusal_does_not_publish_unowned_text() {
        let budget=Breakers::leak_for_tests(1);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("telemetry denial",budget);
        let result=memory::scoped(Some(scope.clone()),||RecoveryText::try_new("node-id"));
        assert_eq!(result.unwrap_err().status,429);
        drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
        assert_eq!(budget.get(Kind::Accounting).used(),0);
    }
}
