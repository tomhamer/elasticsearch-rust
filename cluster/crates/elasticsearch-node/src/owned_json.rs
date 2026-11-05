//! JSON trees admitted before owned strings/container growth. The tree cannot
//! be moved out independently of its reservation; queued work owns this value.
use elasticsearch_engine::{error::EsError, memory::{Breakers, Reservation}};
use serde::de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

pub(crate) struct OwnedJson { value: Value, replies: Vec<(Reservation, Option<crate::replication::decode::DecodeMemory>)>, wire: Option<crate::replication::decode::DecodeMemory>, _memory: Reservation }
impl std::fmt::Debug for OwnedJson { fn fmt(&self,f:&mut std::fmt::Formatter)->std::fmt::Result { std::fmt::Debug::fmt(&self.value,f) } }
impl std::ops::Deref for OwnedJson { type Target=Value; fn deref(&self)->&Value { &self.value } }
impl serde::Serialize for OwnedJson {fn serialize<S:serde::Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{serde::Serialize::serialize(&self.value,serializer)}}
impl OwnedJson {
    /// Synchronous terminal projection. The returned value must not retain the
    /// consumed tree; queued consumers keep this OwnedJson until this callback.
    pub(crate) fn consume(self, f: impl FnOnce(Value, &mut Reservation) -> Result<(), EsError>) -> Result<(), EsError> {
        let Self {value, mut _memory, wire, replies}=self;
        let _retained=(wire,replies);
        f(value,&mut _memory)
    }
    pub(crate) fn from_reserved(value:Value,memory:Reservation)->Self{Self{value,_memory:memory.into_result_owner(),wire:None,replies:Vec::new()}}
    pub(crate) fn reserve_extra(&mut self, bytes: usize, label: &str) -> Result<(), EsError> { self._memory.grow(bytes as u64,label) }
    /// Deduplicating the JSON does not finish the incoming reply channels.
    /// Keep their original reservations (and request scopes) until this job's
    /// replies finish. Tree charges are conservatively retained as well.
    pub(crate) fn prepare_reply_owner(&mut self, other: &Self) -> Result<(), EsError> {
        let required=self.replies.len().checked_add(other.replies.len()).and_then(|n|n.checked_add(1))
            .ok_or_else(||EsError::new(429,"circuit_breaking_exception","mapping reply owner capacity overflow"))?;
        if required>self.replies.capacity() {
            let bytes=required.checked_mul(std::mem::size_of::<(Reservation,Option<crate::replication::decode::DecodeMemory>)>())
                .ok_or_else(||EsError::new(429,"circuit_breaking_exception","mapping reply owner byte overflow"))?;
            self.reserve_extra(bytes,"mapping reply owner replacement")?;
            self.replies.try_reserve_exact(required-self.replies.len())
                .map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        }
        Ok(())
    }
    pub(crate) fn retain_reply_owner(&mut self, other: Self) {
        debug_assert!(self.replies.capacity() >= self.replies.len()+other.replies.len()+1);
        let Self{value,_memory,wire,mut replies}=other;
        drop(value);
        self.replies.append(&mut replies);
        self.replies.push((_memory,wire));
    }
    /// Transform retained JSON while charging new owned storage before growth.
    pub(crate) fn transform(&mut self, f: impl FnOnce(&mut Value,&mut Reservation)->Result<(),EsError>)->Result<(),EsError> { f(&mut self.value,&mut self._memory) }

    pub(crate) fn attach_wire_owner(mut self, owner: crate::replication::decode::DecodeMemory) -> Self { self.wire=Some(owner); self }
    pub(crate) fn parse(text:&str)->Result<Self,EsError> { Self::parse_with(text,elasticsearch_engine::memory::breakers()) }
    pub(crate) fn parse_with(text:&str,budget:&'static Breakers)->Result<Self,EsError> {
        // During geometric unescape scratch growth, old and replacement buffers
        // coexist. Four times input plus minimum-capacity slack covers both;
        // this transient parser storage is separate from retained tree data.
        let _scratch=budget.reserve_request((text.len() as u64).saturating_mul(4).saturating_add(256),"JSON parser escape scratch")?;
        let mut memory=budget.reserve_request(0,"owned JSON tree")?;
        let mut failure=None;
        let mut parser=serde_json::Deserializer::from_str(text);
        let result=Seed{memory:&mut memory,failure:&mut failure}.deserialize(&mut parser);
        if let Some(error)=failure { return Err(error); }
        let value=result.map_err(|e|EsError::new(400,"mapper_parsing_exception",e.to_string()))?;
        parser.end().map_err(|e|EsError::new(400,"mapper_parsing_exception",e.to_string()))?;
        // Parsed results may outlive the decoding request; detach only its
        // scope bookkeeping, retaining the actual budget through tree lifetime.
        Ok(Self{value,_memory:memory.into_result_owner(),wire:None,replies:Vec::new()})
    }
}
struct Seed<'a> { memory:&'a mut Reservation, failure:&'a mut Option<EsError> }
impl Seed<'_> {
    fn charge<E:Error>(&mut self,bytes:usize)->Result<(),E> {
        self.memory.grow(bytes as u64,"owned JSON allocation").map_err(|e| {
            let message=e.to_string(); *self.failure=Some(e); E::custom(message)
        })
    }
    fn allocation_error<E:Error>(&mut self,error:impl std::fmt::Display)->E {
        let message=error.to_string();
        *self.failure=Some(EsError::new(429,"circuit_breaking_exception",message.clone()));
        E::custom(message)
    }
    fn child(&mut self)->Seed<'_> { Seed{memory:self.memory,failure:self.failure} }
}
impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value=Value;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Value,D::Error> { d.deserialize_any(self) }
}
impl<'de> Visitor<'de> for Seed<'_> {
    type Value=Value;
    fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result { f.write_str("JSON value") }
    fn visit_unit<E:Error>(self)->Result<Value,E>{Ok(Value::Null)}
    fn visit_bool<E:Error>(self,v:bool)->Result<Value,E>{Ok(v.into())}
    fn visit_i64<E:Error>(self,v:i64)->Result<Value,E>{Ok(v.into())}
    fn visit_u64<E:Error>(self,v:u64)->Result<Value,E>{Ok(v.into())}
    fn visit_f64<E:Error>(self,v:f64)->Result<Value,E>{Ok(serde_json::Number::from_f64(v).map_or(Value::Null,Value::Number))}
    fn visit_str<E:Error>(mut self,v:&str)->Result<Value,E>{
        self.charge(v.len())?;
        let mut out=String::new();out.try_reserve_exact(v.len()).map_err(|e|self.allocation_error::<E>(e))?;out.push_str(v);
        Ok(Value::String(out))
    }
    fn visit_borrowed_str<E:Error>(self,v:&'de str)->Result<Value,E>{self.visit_str(v)}
    fn visit_seq<A:SeqAccess<'de>>(mut self,mut seq:A)->Result<Value,A::Error>{
        let mut out=Vec::new();
        while let Some(value)=seq.next_element_seed(self.child())? {
            if out.len()==out.capacity() {
                let capacity=out.capacity().saturating_mul(2).max(4);
                self.charge(capacity.saturating_mul(std::mem::size_of::<Value>()))?;
                out.try_reserve_exact(capacity-out.len()).map_err(|e|self.allocation_error::<A::Error>(e))?;
            }
            out.push(value);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A:MapAccess<'de>>(mut self,mut map:A)->Result<Value,A::Error>{
        let mut out=serde_json::Map::new();
        while let Some(key)=map.next_key_seed(Key(self.child()))? {
            // Cover both serde_json Map backends: a BTreeMap node split,
            // or preserve_order IndexMap's entry/hash-table geometric growth
            // including old/new overlap. Sixteen entry+pointer units per
            // insertion exceed the live-plus-replacement capacities of both
            // layouts; key bytes are separately admitted. Duplicate inserts
            // conservatively keep their charge through the tree lifetime.
            self.charge(16usize.saturating_mul(std::mem::size_of::<(String,Value)>()+std::mem::size_of::<usize>()))?;
            let value=map.next_value_seed(self.child())?;
            out.insert(key,value);
        }
        Ok(Value::Object(out))
    }
}
struct Key<'a>(Seed<'a>);
impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value=String;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<String,D::Error>{
        struct V<'a>(Seed<'a>);
        impl<'de> Visitor<'de> for V<'_>{
            type Value=String;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("JSON object key")}
            fn visit_str<E:Error>(mut self,v:&str)->Result<String,E>{self.0.charge(v.len())?;let mut s=String::new();s.try_reserve_exact(v.len()).map_err(|e|self.0.allocation_error::<E>(e))?;s.push_str(v);Ok(s)}
            fn visit_borrowed_str<E:Error>(self,v:&'de str)->Result<String,E>{self.visit_str(v)}
        }
        d.deserialize_str(V(self.0))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deduplicated_reply_owners_survive_incoming_tree_and_input() {
        let first_budget=Breakers::leak_for_tests(1<<20);
        let incoming_budget=Breakers::leak_for_tests(1<<20);
        for budget in [first_budget,incoming_budget] {
            budget.apply_settings(&|key| (key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        }
        let mut first=OwnedJson::parse_with(r#"{"field":"keyword"}"#,first_budget).unwrap();
        let mut incoming=OwnedJson::parse_with(r#"{"field":"keyword"}"#,incoming_budget).unwrap();
        incoming.reserve_extra(512,"test reply channel").unwrap();
        let charged=incoming_budget.get(elasticsearch_engine::memory::Kind::Request).used();
        first.prepare_reply_owner(&incoming).unwrap();
        first.retain_reply_owner(incoming);
        assert_eq!(incoming_budget.get(elasticsearch_engine::memory::Kind::Request).used(),charged);
        drop(first);
        assert_eq!(incoming_budget.get(elasticsearch_engine::memory::Kind::Request).used(),0);
        assert_eq!(first_budget.get(elasticsearch_engine::memory::Kind::Request).used(),0);
    }
    #[test]
    fn admission_precedes_tree_growth_and_owner_survives_input() {
        let budget=Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let input=r#"{"a":[1,true,null,"line\nλ"],"a":{"last":2}}"#.to_string();
        let expected:Value=serde_json::from_str(&input).unwrap();
        let tree=OwnedJson::parse_with(&input,budget).unwrap();drop(input);
        assert_eq!(*tree,expected);
        assert!(budget.get(elasticsearch_engine::memory::Kind::Request).used()>0);
        drop(tree);assert_eq!(budget.get(elasticsearch_engine::memory::Kind::Request).used(),0);
        let tiny=Breakers::leak_for_tests(64);
        tiny.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        assert_eq!(OwnedJson::parse_with(r#"{"a":1}"#,tiny).unwrap_err().status,429);
        assert_eq!(tiny.get(elasticsearch_engine::memory::Kind::Request).used(),0);
    }
}
