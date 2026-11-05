//! File recovery scalar controls, wire-compatible with the original structs.
use serde::{Serialize,Deserialize};
use serde::de::{Visitor,SeqAccess,MapAccess,Error};
use super::super::decode::{Text,DecodeMemory};

#[derive(Debug,Serialize)]
pub struct FileRecoveryStartReq {
pub index: String,
pub shard: u32,
pub target_allocation_id: String,
pub target_node_id: String,
#[serde(skip)] _memory:DecodeMemory,
}
impl FileRecoveryStartReq {
#[allow(clippy::too_many_arguments)]
pub fn try_new(index: &str, shard: u32, target_allocation_id: &str, target_node_id: &str)->Result<Self,elasticsearch_engine::error::EsError>{
let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index,target_allocation_id,target_node_id])?;
Ok(Self{index: index.to_owned(),shard: shard,target_allocation_id: target_allocation_id.to_owned(),target_node_id: target_node_id.to_owned(),_memory})
}
}
impl<'de> Deserialize<'de> for FileRecoveryStartReq {
fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
struct V;
impl<'de> Visitor<'de> for V {
type Value=FileRecoveryStartReq;
fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("FileRecoveryStartReq")}
fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"file recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
let target_node_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_node_id"))?;
Ok(FileRecoveryStartReq{index,shard,target_allocation_id,target_node_id,_memory:DecodeMemory::new(memory)})
}
fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
#[derive(Deserialize)] #[serde(field_identifier)] enum Field {
#[serde(rename="index")] F0,
#[serde(rename="shard")] F1,
#[serde(rename="target_allocation_id")] F2,
#[serde(rename="target_node_id")] F3,
#[serde(other)] Ignore,
}
let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"file recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let mut index=None;
let mut shard=None;
let mut target_allocation_id=None;
let mut target_node_id=None;
while let Some(field)=map.next_key::<Field>()? {match field {
Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
Field::F2=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F3=>{if target_node_id.is_some(){return Err(A::Error::duplicate_field("target_node_id"));} target_node_id=Some(map.next_value_seed(Text(&mut memory))?);},
Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
}}
Ok(FileRecoveryStartReq{
index:index.ok_or_else(||A::Error::missing_field("index"))?,
shard:shard.ok_or_else(||A::Error::missing_field("shard"))?,
target_allocation_id:target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
target_node_id:target_node_id.ok_or_else(||A::Error::missing_field("target_node_id"))?,
 _memory:DecodeMemory::new(memory),
})
}
}
d.deserialize_struct("FileRecoveryStartReq", &["index","shard","target_allocation_id","target_node_id"],V)
}
}

#[derive(Debug,Serialize)]
pub struct FileChunkReq {
pub index: String,
pub shard: u32,
pub recovery_id: u64,
pub target_allocation_id: String,
pub source_allocation_id: String,
pub primary_term: u64,
pub path: String,
pub offset: u64,
pub len: u64,
#[serde(skip)] _memory:DecodeMemory,
}
impl FileChunkReq {
#[allow(clippy::too_many_arguments)]
pub fn try_new(index: &str, shard: u32, recovery_id: u64, target_allocation_id: &str, source_allocation_id: &str, primary_term: u64, path: &str, offset: u64, len: u64)->Result<Self,elasticsearch_engine::error::EsError>{
let _memory=DecodeMemory::reserve_texts(elasticsearch_engine::memory::breakers(), &[index,target_allocation_id,source_allocation_id,path])?;
Ok(Self{index: index.to_owned(),shard: shard,recovery_id: recovery_id,target_allocation_id: target_allocation_id.to_owned(),source_allocation_id: source_allocation_id.to_owned(),primary_term: primary_term,path: path.to_owned(),offset: offset,len: len,_memory})
}
}
impl<'de> Deserialize<'de> for FileChunkReq {
fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
struct V;
impl<'de> Visitor<'de> for V {
type Value=FileChunkReq;
fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("FileChunkReq")}
fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"file recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let index=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("index"))?;
let shard=seq.next_element()?.ok_or_else(||A::Error::missing_field("shard"))?;
let recovery_id=seq.next_element()?.ok_or_else(||A::Error::missing_field("recovery_id"))?;
let target_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("target_allocation_id"))?;
let source_allocation_id=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("source_allocation_id"))?;
let primary_term=seq.next_element()?.ok_or_else(||A::Error::missing_field("primary_term"))?;
let path=seq.next_element_seed(Text(&mut memory))?.ok_or_else(||A::Error::missing_field("path"))?;
let offset=seq.next_element()?.ok_or_else(||A::Error::missing_field("offset"))?;
let len=seq.next_element()?.ok_or_else(||A::Error::missing_field("len"))?;
Ok(FileChunkReq{index,shard,recovery_id,target_allocation_id,source_allocation_id,primary_term,path,offset,len,_memory:DecodeMemory::new(memory)})
}
fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
#[derive(Deserialize)] #[serde(field_identifier)] enum Field {
#[serde(rename="index")] F0,
#[serde(rename="shard")] F1,
#[serde(rename="recovery_id")] F2,
#[serde(rename="target_allocation_id")] F3,
#[serde(rename="source_allocation_id")] F4,
#[serde(rename="primary_term")] F5,
#[serde(rename="path")] F6,
#[serde(rename="offset")] F7,
#[serde(rename="len")] F8,
#[serde(other)] Ignore,
}
let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"file recovery control").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
let mut index=None;
let mut shard=None;
let mut recovery_id=None;
let mut target_allocation_id=None;
let mut source_allocation_id=None;
let mut primary_term=None;
let mut path=None;
let mut offset=None;
let mut len=None;
while let Some(field)=map.next_key::<Field>()? {match field {
Field::F0=>{if index.is_some(){return Err(A::Error::duplicate_field("index"));} index=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F1=>{if shard.is_some(){return Err(A::Error::duplicate_field("shard"));} shard=Some(map.next_value()?);},
Field::F2=>{if recovery_id.is_some(){return Err(A::Error::duplicate_field("recovery_id"));} recovery_id=Some(map.next_value()?);},
Field::F3=>{if target_allocation_id.is_some(){return Err(A::Error::duplicate_field("target_allocation_id"));} target_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F4=>{if source_allocation_id.is_some(){return Err(A::Error::duplicate_field("source_allocation_id"));} source_allocation_id=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F5=>{if primary_term.is_some(){return Err(A::Error::duplicate_field("primary_term"));} primary_term=Some(map.next_value()?);},
Field::F6=>{if path.is_some(){return Err(A::Error::duplicate_field("path"));} path=Some(map.next_value_seed(Text(&mut memory))?);},
Field::F7=>{if offset.is_some(){return Err(A::Error::duplicate_field("offset"));} offset=Some(map.next_value()?);},
Field::F8=>{if len.is_some(){return Err(A::Error::duplicate_field("len"));} len=Some(map.next_value()?);},
Field::Ignore=>{let _:serde::de::IgnoredAny=map.next_value()?;},
}}
Ok(FileChunkReq{
index:index.ok_or_else(||A::Error::missing_field("index"))?,
shard:shard.ok_or_else(||A::Error::missing_field("shard"))?,
recovery_id:recovery_id.ok_or_else(||A::Error::missing_field("recovery_id"))?,
target_allocation_id:target_allocation_id.ok_or_else(||A::Error::missing_field("target_allocation_id"))?,
source_allocation_id:source_allocation_id.ok_or_else(||A::Error::missing_field("source_allocation_id"))?,
primary_term:primary_term.ok_or_else(||A::Error::missing_field("primary_term"))?,
path:path.ok_or_else(||A::Error::missing_field("path"))?,
offset:offset.ok_or_else(||A::Error::missing_field("offset"))?,
len:len.ok_or_else(||A::Error::missing_field("len"))?,
 _memory:DecodeMemory::new(memory),
})
}
}
d.deserialize_struct("FileChunkReq", &["index","shard","recovery_id","target_allocation_id","source_allocation_id","primary_term","path","offset","len"],V)
}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize)]
    struct OldChunk {index:String,shard:u32,recovery_id:u64,target_allocation_id:String,source_allocation_id:String,primary_term:u64,path:String,offset:u64,len:u64}
    #[test]
    fn file_control_preserves_old_chunk_wire_and_owns_each_text() {
        let old=OldChunk{index:"i".into(),shard:2,recovery_id:3,target_allocation_id:"target".into(),source_allocation_id:"source".into(),primary_term:4,path:"seg_1.rstc".into(),offset:5,len:6};
        let bytes=bincode::serialize(&old).unwrap();
        let decoded:FileChunkReq=bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded._memory.bytes(),23);
        assert_eq!(bincode::serialize(&decoded).unwrap(),bytes);
        let json=serde_json::to_string(&old).unwrap();
        let from_json:FileChunkReq=serde_json::from_str(&json).unwrap();
        assert_eq!(bincode::serialize(&from_json).unwrap(),bytes);
        drop(bytes);drop(json);drop(old);
        assert_eq!(decoded.path,"seg_1.rstc");
    }
}
