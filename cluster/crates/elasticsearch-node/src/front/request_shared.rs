//! Shared auxiliary request payloads. Local request edits never duplicate
//! potentially large DFS statistics or application plans.
use super::admitted;
use elasticsearch_engine::{error::{Result,EsError},memory::{breakers,Reservation}};
use elasticsearch_http::{DfsStats,DfsTerm,DfsField};
use serde::{Serialize,Serializer,Deserialize};
use std::{sync::Arc,ops::Deref};
#[derive(Debug)]
struct Owned<T>{value:T,_memory:Reservation}
#[derive(Clone,Debug)]
pub struct SharedDfsStats(elasticsearch_engine::search::phases::AdmittedDfsStats);
impl Deref for SharedDfsStats{type Target=DfsStats;fn deref(&self)->&DfsStats{&self.0}}
impl PartialEq for SharedDfsStats{fn eq(&self,other:&Self)->bool{self.0==other.0}}
impl Serialize for SharedDfsStats{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{self.0.serialize(s)}}
impl SharedDfsStats {
 pub fn try_new(stats:&DfsStats)->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared DFS stats")?;
  let mut terms=admitted::slots(&mut memory,stats.terms.len())?;
  for t in &stats.terms{terms.push(DfsTerm{field:admitted::text(&mut memory,&t.field)?,term:admitted::bytes(&mut memory,&t.term)?,doc_freq:t.doc_freq,total_term_freq:t.total_term_freq,field_doc_count:t.field_doc_count});}
  let mut fields=admitted::slots(&mut memory,stats.fields.len())?;
  for f in &stats.fields{fields.push(DfsField{field:admitted::text(&mut memory,&f.field)?,doc_count:f.doc_count,sum_total_term_freq:f.sum_total_term_freq});}
  Self::from_reserved(DfsStats{terms,fields},memory)
 }
 pub(super) fn from_reserved(value:DfsStats,memory:Reservation)->Result<Self>{elasticsearch_engine::search::phases::AdmittedDfsStats::from_reserved(value,memory).map(Self)}
 pub(super) fn engine_owner(&self)->elasticsearch_engine::search::phases::AdmittedDfsStats{self.0.clone()}

}
#[derive(Debug,Serialize,PartialEq)]
pub struct PlanData{pub indices:Vec<String>,#[serde(with="super::wire_bytes")]pub query_json:Vec<u8>}
#[derive(Clone,Debug)]
pub struct AppSearchPlan(Arc<Owned<PlanData>>);
impl Deref for AppSearchPlan{type Target=PlanData;fn deref(&self)->&PlanData{&self.0.value}}
impl PartialEq for AppSearchPlan{fn eq(&self,other:&Self)->bool{self.0.value==other.0.value}}
impl Serialize for AppSearchPlan{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{self.0.value.serialize(s)}}
impl AppSearchPlan {
 pub fn try_json(indices:&[String],query:&serde_json::Value)->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared application JSON plan")?;
  let mut copied=admitted::slots(&mut memory,indices.len())?;
  for index in indices{copied.push(admitted::text(&mut memory,index)?);}
  let query_json=admitted::json(&mut memory,query)?;
  Self::from_reserved(PlanData{indices:copied,query_json},memory)
 }
 pub fn try_new(indices:&[String],query_json:&[u8])->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared application plan")?;
  let mut copied=admitted::slots(&mut memory,indices.len())?;
  for index in indices{copied.push(admitted::text(&mut memory,index)?);}
  let query_json=admitted::bytes(&mut memory,query_json)?;
  Self::from_reserved(PlanData{indices:copied,query_json},memory)
 }
 pub(super) fn from_reserved(value:PlanData,mut memory:Reservation)->Result<Self>{memory.grow((std::mem::size_of::<Owned<PlanData>>()+2*std::mem::size_of::<usize>())as u64,"application plan owner")?;Ok(Self(Arc::new(Owned{value,_memory:memory.into_result_owner()})))}
}
impl<'de> Deserialize<'de> for SharedDfsStats{fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{super::decode::shared_dfs(d)}}
impl<'de> Deserialize<'de> for AppSearchPlan{fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{super::decode::shared_plan(d)}}
#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn dfs_engine_handoff_keeps_strings_admitted_after_request_wrapper_drops(){
  use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
  let budget=Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("DFS handoff",budget);
  let source=DfsStats{terms:vec![DfsTerm{field:"f".into(),term:vec![42;4096],doc_freq:1,total_term_freq:1,field_doc_count:1}],fields:vec![]};
  let stats=scoped(Some(scope.clone()),||SharedDfsStats::try_new(&source)).unwrap();
  let engine=stats.engine_owner();let copy=engine.clone();
  assert_eq!(engine.terms.as_ptr(),stats.terms.as_ptr());drop(stats);drop(scope);drop(source);
  assert!(budget.get(Kind::Request).used()>=4096);drop(engine);assert_eq!(copy.terms[0].term.len(),4096);
  drop(copy);assert_eq!(budget.get(Kind::Request).used(),0);
 }
}

/// Credentials retained by callbacks, background jobs and internal wire
/// messages. Cloning shares the admitted strings rather than copying them.
#[derive(Clone,Debug,Default)]
pub struct SharedHeaders(Option<Arc<Owned<Vec<(String,String)>>>>);
impl Deref for SharedHeaders {type Target=[(String,String)];fn deref(&self)->&Self::Target{self.0.as_ref().map_or(&[],|v|v.value.as_slice())}}
impl Serialize for SharedHeaders {fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{self.deref().serialize(s)}}
impl<'de> Deserialize<'de> for SharedHeaders {fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{super::decode::shared_headers(d)}}
impl SharedHeaders {
 pub fn try_pairs(headers:&[(&str,&str)])->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared borrowed headers")?;
  let mut value=admitted::slots(&mut memory,headers.len())?;
  for (key,text) in headers {value.push((admitted::text(&mut memory,key)?,admitted::text(&mut memory,text)?));}
  Self::from_reserved(value,memory)
 }
 pub fn try_filtered(headers:&[(String,String)],keep:impl Fn(&(String,String))->bool)->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared credential headers")?;
  let mut value=admitted::slots(&mut memory,headers.len())?;
  for pair in headers {if keep(pair){value.push((admitted::text(&mut memory,&pair.0)?,admitted::text(&mut memory,&pair.1)?));}}
  Self::from_reserved(value,memory)
 }
 pub fn try_new(headers:&[(String,String)])->Result<Self>{Self::try_filtered(headers, |_|true)}
 pub(super) fn from_reserved(value:Vec<(String,String)>,mut memory:Reservation)->Result<Self>{
  memory.grow((std::mem::size_of::<Owned<Vec<(String,String)>>>()+2*std::mem::size_of::<usize>())as u64,"shared headers owner")?;
  Ok(Self(Some(Arc::new(Owned{value,_memory:memory.into_result_owner()}))))
 }
}

#[derive(Clone,Debug,Default)]
pub struct SharedText(Option<Arc<Owned<String>>>);
impl Deref for SharedText {type Target=str;fn deref(&self)->&str{self.0.as_ref().map_or("",|v|v.value.as_str())}}
impl std::fmt::Display for SharedText {fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(self)}}
impl Serialize for SharedText {fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{str::serialize(self,s)}}
impl<'de> Deserialize<'de> for SharedText {fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{super::decode::shared_text(d)}}
impl SharedText {
 pub fn as_str(&self)->&str{self}

 pub fn try_new(text:&str)->Result<Self>{
  let mut memory=breakers().reserve_request(0,"shared text")?;
  let value=admitted::text(&mut memory,text)?;Self::from_reserved(value,memory)
 }
 pub fn try_json<T:Serialize>(value:&T)->Result<Self>{
  let bytes=admitted::encoded_body(|out|serde_json::to_writer(out,value).map_err(std::io::Error::other),breakers())?;
  let text=std::str::from_utf8(&bytes).map_err(|e|EsError::internal(e.to_string()))?;
  Self::try_new(text)
 }

 pub fn try_format(args:std::fmt::Arguments<'_>)->Result<Self>{
  let bytes=admitted::encoded_body(|out|out.write_fmt(args),breakers())?;
  let text=std::str::from_utf8(&bytes).map_err(|e|EsError::internal(e.to_string()))?;
  let mut memory=breakers().reserve_request(0,"shared text")?;
  let value=admitted::text(&mut memory,text)?;Self::from_reserved(value,memory)
 }
 pub(crate) fn from_reserved(value:String,mut memory:Reservation)->Result<Self>{
  memory.grow((std::mem::size_of::<Owned<String>>()+2*std::mem::size_of::<usize>())as u64,"shared text owner")?;
  Ok(Self(Some(Arc::new(Owned{value,_memory:memory.into_result_owner()}))))
 }
}
/// Percent-encoding without an intermediate String; the destination owns
/// admission before this formatter writes any bytes.
pub(crate) struct UriComponent<'a>(pub &'a str);
impl std::fmt::Display for UriComponent<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
 for b in self.0.bytes(){if b.is_ascii_alphanumeric()||matches!(b,b'-'|b'_'|b'.'|b'~'){write!(f,"{}",b as char)?;}else{write!(f,"%{b:02X}")?;}}Ok(())
}}

/// Borrowed query filtering. Formatting streams directly into an admitted URI.
pub(crate) struct FilteredQuery<'a> { pub query:&'a str, pub drop:&'a [&'a str], pub append:&'a str }
impl std::fmt::Display for FilteredQuery<'_> {
 fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
  let mut sep="?";
  for part in self.query.split('&').filter(|p|!p.is_empty()&&!self.drop.contains(&p.split('=').next().unwrap_or(""))) {
   write!(f,"{sep}{part}")?;sep="&";
  }
  if !self.append.is_empty(){write!(f,"{sep}{}",self.append)?;}
  Ok(())
 }
}
/// Cloning an iterator retains borrowed inputs, never allocates a joined string.
pub(crate) struct UriJoin<I>(pub I,pub &'static str);
impl<I> std::fmt::Display for UriJoin<I> where I:Iterator+Clone,I::Item:std::fmt::Display {
 fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
  for (i,item) in self.0.clone().enumerate(){if i>0 {f.write_str(self.1)?;}write!(f,"{item}")?;}Ok(())
 }
}

impl PartialEq for SharedText {fn eq(&self,other:&Self)->bool{**self==**other}}
impl Eq for SharedText {}
impl PartialEq<str> for SharedText {fn eq(&self,other:&str)->bool{&**self==other}}
impl PartialEq<String> for SharedText {fn eq(&self,other:&String)->bool{&**self==other}}
impl PartialEq<&str> for SharedText {fn eq(&self,other:&&str)->bool{&**self==*other}}
impl PartialEq<SharedText> for String {fn eq(&self,other:&SharedText)->bool{self.as_str()==&**other}}
impl AsRef<str> for SharedText {fn as_ref(&self)->&str{self}}

impl std::hash::Hash for SharedText{fn hash<H:std::hash::Hasher>(&self,state:&mut H){std::hash::Hash::hash(&**self,state)}}
impl std::borrow::Borrow<str> for SharedText{fn borrow(&self)->&str{self}}

impl PartialOrd for SharedText {fn partial_cmp(&self,other:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(other))}}
impl Ord for SharedText {fn cmp(&self,other:&Self)->std::cmp::Ordering{self.as_str().cmp(other.as_str())}}
