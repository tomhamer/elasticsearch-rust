//! Recovery control envelopes retain pre-copy scalar admission through async RPC.
//! Field order and serialized representation are unchanged.
use serde::{Serialize,Deserialize};
use serde::de::{Visitor,SeqAccess,MapAccess,Error};
use crate::replication::decode::{Text,OptionalText,DecodeMemory};
use crate::replication::replica::ReplicaPrefix;

#[derive(Serialize)]
pub struct RecoveryStartReq {
    pub index: String,
    pub shard: u32,
    pub target_allocation_id: String,
    pub from_seq_no: Option<i64>,
    #[serde(skip)]
    _memory: DecodeMemory,
}
impl RecoveryStartReq {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(index: &str, shard: u32, target_allocation_id: &str, from_seq_no: Option<i64>)->Result<Self,elasticsearch_engine::error::EsError> {
        let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index, target_allocation_id])?;
        Ok(Self{index: index.to_owned(), shard: shard, target_allocation_id: target_allocation_id.to_owned(), from_seq_no: from_seq_no,_memory})
    }
}
impl<'de> Deserialize<'de> for RecoveryStartReq {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=RecoveryStartReq;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("RecoveryStartReq")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
                let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
                let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
                let from_seq_no=seq.next_element()?.ok_or_else(||A::Error::missing_field("from_seq_no"))?;
                Ok(RecoveryStartReq{index,shard,target_allocation_id,from_seq_no,_memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(Deserialize)] #[serde(field_identifier)] enum Field {
                    #[serde(rename="index")] F0,
                    #[serde(rename="shard")] F1,
                    #[serde(rename="target_allocation_id")] F2,
                    #[serde(rename="from_seq_no")] F3,
                    #[serde(other)] Ignore,
                }
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index=None;
                let mut shard=None;
                let mut target_allocation_id=None;
                let mut from_seq_no=None;
                while let Some(field)=map.next_key::<Field>()? {match field {
                    Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
                    Field::F2=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F3=>{if from_seq_no.is_some(){return Err(A::Error::duplicate_field("from_seq_no"));} from_seq_no=Some(map.next_value()?);},
                    Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
                }}
                Ok(RecoveryStartReq{
                    index: index.ok_or_else(||A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(||A::Error::missing_field("shard"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
                    from_seq_no: from_seq_no.unwrap_or(None),
                    _memory:DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("RecoveryStartReq", &["index","shard","target_allocation_id","from_seq_no"],V)
    }
}

#[derive(Serialize)]
pub struct RecoveryReuseReq {
    pub index: String,
    pub shard: u32,
    pub cluster_uuid: String,
    pub index_uuid: String,
    pub old_allocation_id: String,
    pub target_allocation_id: String,
    pub target_node_id: String,
    pub source_allocation_id: String,
    pub primary_term: u64,
    pub prefix: ReplicaPrefix,
    pub history_uuid: Option<String>,
    #[serde(skip)]
    _memory: DecodeMemory,
}
impl RecoveryReuseReq {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(index: &str, shard: u32, cluster_uuid: &str, index_uuid: &str, old_allocation_id: &str, target_allocation_id: &str, target_node_id: &str, source_allocation_id: &str, primary_term: u64, prefix: ReplicaPrefix, history_uuid: Option<&str>)->Result<Self,elasticsearch_engine::error::EsError> {
        let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index, cluster_uuid, index_uuid, old_allocation_id, target_allocation_id, target_node_id, source_allocation_id, history_uuid.unwrap_or("")])?;
        Ok(Self{index: index.to_owned(), shard: shard, cluster_uuid: cluster_uuid.to_owned(), index_uuid: index_uuid.to_owned(), old_allocation_id: old_allocation_id.to_owned(), target_allocation_id: target_allocation_id.to_owned(), target_node_id: target_node_id.to_owned(), source_allocation_id: source_allocation_id.to_owned(), primary_term: primary_term, prefix: prefix, history_uuid: history_uuid.map(str::to_owned),_memory})
    }
}
impl<'de> Deserialize<'de> for RecoveryReuseReq {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=RecoveryReuseReq;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("RecoveryReuseReq")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
                let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
                let cluster_uuid=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("cluster_uuid"))?;
                let index_uuid=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index_uuid"))?;
                let old_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("old_allocation_id"))?;
                let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
                let target_node_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_node_id"))?;
                let source_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("source_allocation_id"))?;
                let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
                let prefix=seq.next_element()?.ok_or_else(||A::Error::missing_field("prefix"))?;
                let history_uuid=seq.next_element_seed(OptionalText(&mut memory))?.ok_or_else(||A::Error::missing_field("history_uuid"))?;
                Ok(RecoveryReuseReq{index,shard,cluster_uuid,index_uuid,old_allocation_id,target_allocation_id,target_node_id,source_allocation_id,primary_term,prefix,history_uuid,_memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(Deserialize)] #[serde(field_identifier)] enum Field {
                    #[serde(rename="index")] F0,
                    #[serde(rename="shard")] F1,
                    #[serde(rename="cluster_uuid")] F2,
                    #[serde(rename="index_uuid")] F3,
                    #[serde(rename="old_allocation_id")] F4,
                    #[serde(rename="target_allocation_id")] F5,
                    #[serde(rename="target_node_id")] F6,
                    #[serde(rename="source_allocation_id")] F7,
                    #[serde(rename="primary_term")] F8,
                    #[serde(rename="prefix")] F9,
                    #[serde(rename="history_uuid")] F10,
                    #[serde(other)] Ignore,
                }
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index=None;
                let mut shard=None;
                let mut cluster_uuid=None;
                let mut index_uuid=None;
                let mut old_allocation_id=None;
                let mut target_allocation_id=None;
                let mut target_node_id=None;
                let mut source_allocation_id=None;
                let mut primary_term=None;
                let mut prefix=None;
                let mut history_uuid=None;
                while let Some(field)=map.next_key::<Field>()? {match field {
                    Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
                    Field::F2=>{if cluster_uuid.is_some(){return Err(A::Error::duplicate_field("cluster_uuid"));} cluster_uuid=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F3=>{if index_uuid.is_some(){return Err(A::Error::duplicate_field("index_uuid"));} index_uuid=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F4=>{if old_allocation_id.is_some(){return Err(A::Error::duplicate_field("old_allocation_id"));} old_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F5=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F6=>{if target_node_id.is_some(){return Err(A::Error::duplicate_field("target_node_id"));} target_node_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F7=>{if source_allocation_id.is_some(){return Err(A::Error::duplicate_field("source_allocation_id"));} source_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F8=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));} primary_term=Some(map.next_value()?);},
                    Field::F9=>{if prefix.is_some(){return Err(A::Error::duplicate_field("prefix"));} prefix=Some(map.next_value()?);},
                    Field::F10=>{if history_uuid.is_some(){return Err(A::Error::duplicate_field("history_uuid"));} history_uuid=Some(map.next_value_seed(OptionalText(&mut memory))?);},
                    Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
                }}
                Ok(RecoveryReuseReq{
                    index: index.ok_or_else(||A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(||A::Error::missing_field("shard"))?,
                    cluster_uuid: cluster_uuid.ok_or_else(||A::Error::missing_field("cluster_uuid"))?,
                    index_uuid: index_uuid.ok_or_else(||A::Error::missing_field("index_uuid"))?,
                    old_allocation_id: old_allocation_id.ok_or_else(||A::Error::missing_field("old_allocation_id"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
                    target_node_id: target_node_id.ok_or_else(||A::Error::missing_field("target_node_id"))?,
                    source_allocation_id: source_allocation_id.ok_or_else(||A::Error::missing_field("source_allocation_id"))?,
                    primary_term: primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?,
                    prefix: prefix.ok_or_else(||A::Error::missing_field("prefix"))?,
                    history_uuid: history_uuid.unwrap_or(None),
                    _memory:DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("RecoveryReuseReq", &["index","shard","cluster_uuid","index_uuid","old_allocation_id","target_allocation_id","target_node_id","source_allocation_id","primary_term","prefix","history_uuid"],V)
    }
}

#[derive(Serialize)]
pub struct RecoveryBatchReq {
    pub index: String,
    pub shard: u32,
    pub recovery_id: u64,
    pub target_allocation_id: String,
    pub target_node_id: String,
    pub target_address: String,
    #[serde(skip)]
    _memory: DecodeMemory,
}
impl RecoveryBatchReq {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(index: &str, shard: u32, recovery_id: u64, target_allocation_id: &str, target_node_id: &str, target_address: &str)->Result<Self,elasticsearch_engine::error::EsError> {
        let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index, target_allocation_id, target_node_id, target_address])?;
        Ok(Self{index: index.to_owned(), shard: shard, recovery_id: recovery_id, target_allocation_id: target_allocation_id.to_owned(), target_node_id: target_node_id.to_owned(), target_address: target_address.to_owned(),_memory})
    }
}
impl<'de> Deserialize<'de> for RecoveryBatchReq {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=RecoveryBatchReq;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("RecoveryBatchReq")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
                let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
                let recovery_id=seq.next_element()?.ok_or_else(||A::Error::missing_field("recovery_id"))?;
                let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
                let target_node_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_node_id"))?;
                let target_address=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_address"))?;
                Ok(RecoveryBatchReq{index,shard,recovery_id,target_allocation_id,target_node_id,target_address,_memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(Deserialize)] #[serde(field_identifier)] enum Field {
                    #[serde(rename="index")] F0,
                    #[serde(rename="shard")] F1,
                    #[serde(rename="recovery_id")] F2,
                    #[serde(rename="target_allocation_id")] F3,
                    #[serde(rename="target_node_id")] F4,
                    #[serde(rename="target_address")] F5,
                    #[serde(other)] Ignore,
                }
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index=None;
                let mut shard=None;
                let mut recovery_id=None;
                let mut target_allocation_id=None;
                let mut target_node_id=None;
                let mut target_address=None;
                while let Some(field)=map.next_key::<Field>()? {match field {
                    Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
                    Field::F2=>{if recovery_id.is_some(){return Err(A::Error::duplicate_field("recovery_id"));} recovery_id=Some(map.next_value()?);},
                    Field::F3=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F4=>{if target_node_id.is_some(){return Err(A::Error::duplicate_field("target_node_id"));} target_node_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F5=>{if target_address.is_some(){return Err(A::Error::duplicate_field("target_address"));} target_address=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
                }}
                Ok(RecoveryBatchReq{
                    index: index.ok_or_else(||A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(||A::Error::missing_field("shard"))?,
                    recovery_id: recovery_id.ok_or_else(||A::Error::missing_field("recovery_id"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
                    target_node_id: target_node_id.ok_or_else(||A::Error::missing_field("target_node_id"))?,
                    target_address: target_address.ok_or_else(||A::Error::missing_field("target_address"))?,
                    _memory:DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("RecoveryBatchReq", &["index","shard","recovery_id","target_allocation_id","target_node_id","target_address"],V)
    }
}

#[derive(Serialize)]
pub struct RecoveryFinalizeReq {
    pub index: String,
    pub shard: u32,
    pub recovery_id: u64,
    pub target_allocation_id: String,
    pub local_checkpoint: i64,
    pub relocation: bool,
    #[serde(skip)]
    _memory: DecodeMemory,
}
impl RecoveryFinalizeReq {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(index: &str, shard: u32, recovery_id: u64, target_allocation_id: &str, local_checkpoint: i64, relocation: bool)->Result<Self,elasticsearch_engine::error::EsError> {
        let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index, target_allocation_id])?;
        Ok(Self{index: index.to_owned(), shard: shard, recovery_id: recovery_id, target_allocation_id: target_allocation_id.to_owned(), local_checkpoint: local_checkpoint, relocation: relocation,_memory})
    }
}
impl<'de> Deserialize<'de> for RecoveryFinalizeReq {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=RecoveryFinalizeReq;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("RecoveryFinalizeReq")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
                let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
                let recovery_id=seq.next_element()?.ok_or_else(||A::Error::missing_field("recovery_id"))?;
                let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
                let local_checkpoint=seq.next_element()?.ok_or_else(||A::Error::missing_field("local_checkpoint"))?;
                let relocation=seq.next_element()?.ok_or_else(||A::Error::missing_field("relocation"))?;
                Ok(RecoveryFinalizeReq{index,shard,recovery_id,target_allocation_id,local_checkpoint,relocation,_memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(Deserialize)] #[serde(field_identifier)] enum Field {
                    #[serde(rename="index")] F0,
                    #[serde(rename="shard")] F1,
                    #[serde(rename="recovery_id")] F2,
                    #[serde(rename="target_allocation_id")] F3,
                    #[serde(rename="local_checkpoint")] F4,
                    #[serde(rename="relocation")] F5,
                    #[serde(other)] Ignore,
                }
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut index=None;
                let mut shard=None;
                let mut recovery_id=None;
                let mut target_allocation_id=None;
                let mut local_checkpoint=None;
                let mut relocation=None;
                while let Some(field)=map.next_key::<Field>()? {match field {
                    Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
                    Field::F2=>{if recovery_id.is_some(){return Err(A::Error::duplicate_field("recovery_id"));} recovery_id=Some(map.next_value()?);},
                    Field::F3=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::F4=>{if local_checkpoint.is_some(){return Err(A::Error::duplicate_field("local_checkpoint"));} local_checkpoint=Some(map.next_value()?);},
                    Field::F5=>{if relocation.is_some(){return Err(A::Error::duplicate_field("relocation"));} relocation=Some(map.next_value()?);},
                    Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
                }}
                Ok(RecoveryFinalizeReq{
                    index: index.ok_or_else(||A::Error::missing_field("index"))?,
                    shard: shard.ok_or_else(||A::Error::missing_field("shard"))?,
                    recovery_id: recovery_id.ok_or_else(||A::Error::missing_field("recovery_id"))?,
                    target_allocation_id: target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
                    local_checkpoint: local_checkpoint.ok_or_else(||A::Error::missing_field("local_checkpoint"))?,
                    relocation: relocation.ok_or_else(||A::Error::missing_field("relocation"))?,
                    _memory:DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("RecoveryFinalizeReq", &["index","shard","recovery_id","target_allocation_id","local_checkpoint","relocation"],V)
    }
}

#[derive(Serialize)]
pub struct BoundRecoveryReq<T> {
    pub source_allocation_id:String,
    pub primary_term:u64,
    pub request:T,
    #[serde(skip)]
    _memory:DecodeMemory,
}
impl<T> BoundRecoveryReq<T> {
    pub fn try_new(source_allocation_id:&str,primary_term:u64,request:T)->Result<Self,elasticsearch_engine::error::EsError> {
        let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[source_allocation_id])?;
        Ok(Self{source_allocation_id:source_allocation_id.to_owned(),primary_term,request,_memory})
    }
}
impl<'de,T:Deserialize<'de>> Deserialize<'de> for BoundRecoveryReq<T> {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
        struct V<T>(std::marker::PhantomData<T>);
        impl<'de,T:Deserialize<'de>> Visitor<'de> for V<T> {
            type Value=BoundRecoveryReq<T>;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("bound recovery request")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"bound recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let source_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("source_allocation_id"))?;
                let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
                let request=seq.next_element()?.ok_or_else(||A::Error::missing_field("request"))?;
                Ok(BoundRecoveryReq{source_allocation_id,primary_term,request,_memory:DecodeMemory::new(memory)})
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(Deserialize)] #[serde(field_identifier,rename_all="snake_case")]
                enum Field {SourceAllocationId,PrimaryTerm,Request,#[serde(other)]Ignore}
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"bound recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let (mut source_allocation_id,mut primary_term,mut request)=(None,None,None);
                while let Some(field)=map.next_key::<Field>()? {match field {
                    Field::SourceAllocationId=>{if source_allocation_id.is_some(){return Err(A::Error::duplicate_field("source_allocation_id"));} source_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
                    Field::PrimaryTerm=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));} primary_term=Some(map.next_value()?);},
                    Field::Request=>{if request.is_some(){return Err(A::Error::duplicate_field("request"));} request=Some(map.next_value()?);},
                    Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
                }}
                Ok(BoundRecoveryReq{
                    source_allocation_id:source_allocation_id.ok_or_else(||A::Error::missing_field("source_allocation_id"))?,
                    primary_term:primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?,
                    request:request.ok_or_else(||A::Error::missing_field("request"))?,_memory:DecodeMemory::new(memory),
                })
            }
        }
        d.deserialize_struct("BoundRecoveryReq", &["source_allocation_id","primary_term","request"],V(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize,Deserialize)]
    struct OldStart {index:String,shard:u32,target_allocation_id:String,from_seq_no:Option<i64>}
    #[derive(Serialize,Deserialize)]
    struct OldBound {source_allocation_id:String,primary_term:u64,request:OldStart}
    #[test]
    fn recovery_control_preserves_old_wire_and_retains_scalar_owners() {
        let old=OldBound{source_allocation_id:"source".repeat(100),primary_term:9,
            request:OldStart{index:"index".repeat(200),shard:3,target_allocation_id:"target".repeat(100),from_seq_no:Some(41)}};
        let wire=bincode::serialize(&old).unwrap();
        let decoded:BoundRecoveryReq<RecoveryStartReq>=bincode::deserialize(&wire).unwrap();
        assert_eq!(decoded._memory.bytes(),600);
        assert_eq!(decoded.request._memory.bytes(),1600);
        assert_eq!(bincode::serialize(&decoded).unwrap(),wire);
        let json=serde_json::to_string(&old).unwrap();
        let from_json:BoundRecoveryReq<RecoveryStartReq>=serde_json::from_str(&json).unwrap();
        assert_eq!(bincode::serialize(&from_json).unwrap(),wire);
        drop(wire);drop(json);drop(old);
        assert_eq!(decoded.request.index.len(),1000);
        assert_eq!(decoded.request.from_seq_no,Some(41));
    }
    #[test]
    fn control_json_rejects_duplicate_fields_and_keeps_legacy_optional_default() {
        assert!(serde_json::from_str::<RecoveryStartReq>(r#"{"index":"a","index":"b","shard":0,"target_allocation_id":"t"}"#).is_err());
        let request:RecoveryStartReq=serde_json::from_str(r#"{"index":"a","shard":0,"target_allocation_id":"t"}"#).unwrap();
        assert_eq!(request.from_seq_no,None);
        assert_eq!(request._memory.bytes(),2);
    }
}
