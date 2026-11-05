//! Immutable local request ownership. Clones share payloads; mutations build
//! admitted replacements before swapping, so refusal leaves the request intact.
use super::{admitted, Dfs, AppSearchPlan};
use axum::body::Bytes;
use elasticsearch_engine::{error::{EsError,Result},memory::{breakers,Reservation}};
use std::{sync::Arc,ops::Deref};
use serde::ser::{Serialize,Serializer,SerializeStruct};

#[derive(Clone,Debug)]
pub struct RestReq { payload:Arc<Payload>, pub hops:u8 }
#[derive(Debug)]
pub struct Payload {
    pub method:String,
    pub uri:String,
    pub headers:Vec<(String,String)>,
    pub body:Bytes,
    pub dfs:Dfs,
    pub app_plan:Option<AppSearchPlan>,
    /// The body parsed once ([`RestReq::json_ref`]): the coordinator reads
    /// it at several steps of a search, and a kNN body is thousands of floats.
    /// With whether it parsed as plain JSON (`jsonpos::parse_plain`: no
    /// duplicate keys, comments or trailing tokens), so the engine can take
    /// the value instead of parsing the same bytes again.
    /// Shared by the requests derived from this one that keep its body
    /// (header, URI, method, DFS edits), so a shard request reuses the
    /// coordinator's parse.
    pub(crate) parsed:Arc<std::sync::OnceLock<Option<(Arc<serde_json::Value>,bool)>>>,
    _memory:Arc<Reservation>,
}
impl Deref for RestReq {type Target=Payload;fn deref(&self)->&Payload{&self.payload}}
struct Body {bytes:Vec<u8>,_memory:Arc<Reservation>}
impl AsRef<[u8]> for Body {fn as_ref(&self)->&[u8]{&self.bytes}}
fn owner_size()->u64{(std::mem::size_of::<Payload>()+2*std::mem::size_of::<usize>()+std::mem::size_of::<Reservation>()+2*std::mem::size_of::<usize>())as u64}
/// The bytes a request's metadata holds: its owner, method, URI, header
/// slots and header texts (what one admitted copy of them reserves).
fn metadata_bytes<'a>(method:&str,uri:&str,headers:impl Iterator<Item=(&'a str,&'a str)>)->Result<(u64,usize)>{
    let overflow=||EsError::new(429,"circuit_breaking_exception","request metadata size overflow");
    let mut count=0usize;
    let mut bytes=owner_size().checked_add((method.len()+uri.len()) as u64).ok_or_else(overflow)?;
    for(k,v)in headers{
        count+=1;
        let pair=(std::mem::size_of::<(String,String)>()+k.len()).checked_add(v.len()).ok_or_else(overflow)?;
        bytes=bytes.checked_add(pair as u64).ok_or_else(overflow)?;
    }
    Ok((bytes,count))
}
fn owned_text(value:&str)->Result<String>{
    let mut out=String::new();
    out.try_reserve_exact(value.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
    out.push_str(value);
    Ok(out)
}
impl RestReq {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(method:&str,uri:&str,headers:&[(String,String)],body:&[u8],hops:u8,dfs:Dfs,app_plan:Option<AppSearchPlan>)->Result<Self>{
        let body=admitted::http_body(body,breakers())?;
        Self::from_shared(method,uri,headers,body,hops,dfs,app_plan)
    }
    fn from_shared(method:&str,uri:&str,headers:&[(String,String)],body:Bytes,hops:u8,dfs:Dfs,app_plan:Option<AppSearchPlan>)->Result<Self>{
        Self::from_parts(method,uri,headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body,hops,dfs,app_plan)
    }
    /// One admitted copy of the metadata: the whole size is reserved once
    /// (one breaker check), then the texts are copied. It reserves what the
    /// former per-text growth reserved in total, without a breaker round trip
    /// for every header key and value, and without the intermediate header
    /// copies that header edits used to build and copy again.
    #[allow(clippy::too_many_arguments)]
    fn from_parts<'a,I:Iterator<Item=(&'a str,&'a str)>+Clone>(method:&str,uri:&str,headers:I,body:Bytes,hops:u8,dfs:Dfs,app_plan:Option<AppSearchPlan>)->Result<Self>{
        let (bytes,count)=metadata_bytes(method,uri,headers.clone())?;
        let memory=breakers().reserve_request(bytes,"local request metadata")?;
        let method=owned_text(method)?;
        let uri=owned_text(uri)?;
        let mut out=Vec::new();
        out.try_reserve_exact(count).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        for(k,v)in headers{out.push((owned_text(k)?,owned_text(v)?));}
        Ok(Self{hops,payload:Arc::new(Payload{method,uri,headers:out,body,dfs,app_plan,parsed:Default::default(),_memory:Arc::new(memory.into_result_owner())})})
    }
    /// Adopt already admitted decoded/copied buffers. The shared guard also
    /// follows body extraction, so no String/Vec escapes its original owner.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_reserved(method:String,uri:String,headers:Vec<(String,String)>,body:Vec<u8>,hops:u8,dfs:Dfs,app_plan:Option<AppSearchPlan>,mut memory:Reservation)->Result<Self>{
        memory.grow(owner_size()+(std::mem::size_of::<Body>()+4*std::mem::size_of::<usize>())as u64,"local request owner")?;
        let memory=Arc::new(memory.into_result_owner());
        let body=Bytes::from_owner(Body{bytes:body,_memory:memory.clone()});
        Ok(Self{hops,payload:Arc::new(Payload{method,uri,headers,body,dfs,app_plan,parsed:Default::default(),_memory:memory})})
    }
    /// `next`, rebuilt from this request with the same body: it shares this
    /// request's body parse.
    fn keeping_parse(&self,mut next:Self)->Self{
        if let Some(payload)=Arc::get_mut(&mut next.payload){payload.parsed=self.payload.parsed.clone();}
        next
    }
    pub(crate) fn replace_header_json<T:serde::Serialize+?Sized>(&mut self,name:&str,value:&T)->Result<()> {
        let bytes=admitted::encoded_body(|out|serde_json::to_writer(out,value).map_err(std::io::Error::other),breakers())?;
        let text=std::str::from_utf8(&bytes).map_err(|e|EsError::internal(e.to_string()))?;
        self.replace_header(name,text)
    }
    pub fn set_method(&mut self,value:&str)->Result<()> {let next=self.keeping_parse(Self::from_shared(value,&self.uri,&self.headers,self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);*self=next;Ok(())}
    /// Format URI text into fixed admitted storage before the metadata copy.
    pub fn set_uri_fmt(&mut self,args:std::fmt::Arguments<'_>)->Result<()> {
        let bytes=admitted::encoded_body(|out|out.write_fmt(args),breakers())?;
        let uri=std::str::from_utf8(&bytes).map_err(|e|EsError::internal(e.to_string()))?;
        self.set_uri(uri)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_uri(method:&str,uri:std::fmt::Arguments<'_>,headers:&[(String,String)],body:&[u8],hops:u8,dfs:Dfs,app_plan:Option<AppSearchPlan>)->Result<Self>{
        let bytes=admitted::encoded_body(|out|out.write_fmt(uri),breakers())?;
        let uri=std::str::from_utf8(&bytes).map_err(|e|EsError::internal(e.to_string()))?;
        Self::try_new(method,uri,headers,body,hops,dfs,app_plan)
    }
    pub fn set_uri(&mut self,value:&str)->Result<()> {let next=self.keeping_parse(Self::from_shared(&self.method,value,&self.headers,self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);*self=next;Ok(())}
    pub fn set_headers(&mut self,value:&[(String,String)])->Result<()> {let next=self.keeping_parse(Self::from_shared(&self.method,&self.uri,value,self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);*self=next;Ok(())}
    pub fn set_body(&mut self,value:&[u8])->Result<()> {let next=self.try_with_body(value)?;*self=next;Ok(())}
    pub fn set_json(&mut self,value:&serde_json::Value)->Result<()> {
        let mut memory=breakers().reserve_request(0,"local request JSON")?;
        let raw=admitted::json(&mut memory,value)?;
        // Keep temporary serialized bytes admitted until the body copy exists.
        self.set_body(&raw)
    }
    pub fn try_with_body(&self,value:&[u8])->Result<Self>{Self::from_shared(&self.method,&self.uri,&self.headers,admitted::http_body(value,breakers())?,self.hops,self.dfs.clone(),self.app_plan.clone())}
    pub(crate) fn encode_body(encode:impl Fn(&mut dyn std::io::Write)->std::io::Result<()>)->Result<Bytes>{admitted::encoded_body(encode,breakers())}
    pub(crate) fn try_with_encoded_body(&self,encode:impl Fn(&mut dyn std::io::Write)->std::io::Result<()>)->Result<Self>{
        Self::from_shared(&self.method,&self.uri,&self.headers,admitted::encoded_body(encode,breakers())?,self.hops,self.dfs.clone(),self.app_plan.clone())
    }
    pub fn try_without_body(&self)->Result<Self>{Self::from_shared(&self.method,&self.uri,&self.headers,Bytes::new(),self.hops,self.dfs.clone(),self.app_plan.clone())}
    pub fn retain_headers(&mut self,keep:impl Fn(&(String,String))->bool)->Result<()> {
        let keep=&keep;
        let next=self.keeping_parse(Self::from_parts(&self.method,&self.uri,self.headers.iter().filter(move|pair|keep(*pair)).map(|(k,v)|(k.as_str(),v.as_str())),self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);
        *self=next;Ok(())
    }
    pub fn push_header(&mut self,name:&str,value:&str)->Result<()> {
        let next=self.keeping_parse(Self::from_parts(&self.method,&self.uri,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())).chain(std::iter::once((name,value))),self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);
        *self=next;Ok(())
    }
    pub fn set_dfs(&mut self,dfs:Dfs)->Result<()> {let next=self.keeping_parse(Self::from_shared(&self.method,&self.uri,&self.headers,self.body.clone(),self.hops,dfs,self.app_plan.clone())?);*self=next;Ok(())}
    pub fn set_app_plan(&mut self,app_plan:Option<AppSearchPlan>)->Result<()> {let next=self.keeping_parse(Self::from_shared(&self.method,&self.uri,&self.headers,self.body.clone(),self.hops,self.dfs.clone(),app_plan)?);*self=next;Ok(())}
}
impl Serialize for RestReq {fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{
    let mut out=s.serialize_struct("RestReq",7)?;
    out.serialize_field("method",&self.method)?;out.serialize_field("uri",&self.uri)?;out.serialize_field("headers",&self.headers)?;
    struct Wire<'a>(&'a[u8]);impl Serialize for Wire<'_>{fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{s.serialize_bytes(self.0)}}
    out.serialize_field("body",&Wire(&self.body))?;out.serialize_field("hops",&self.hops)?;out.serialize_field("dfs",&self.dfs)?;out.serialize_field("app_plan",&self.app_plan)?;out.end()
}}
impl RestReq {
    pub fn replace_header(&mut self,name:&str,value:&str)->Result<()> {
        let kept=self.headers.iter().filter(|(key,_)|!key.eq_ignore_ascii_case(name)).map(|(k,v)|(k.as_str(),v.as_str()));
        let next=self.keeping_parse(Self::from_parts(&self.method,&self.uri,kept.chain(std::iter::once((name,value))),self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);
        *self=next;Ok(())
    }
    pub fn set_shard(&mut self,shard:u32)->Result<()> {self.replace_header(crate::shard_engines::SHARD_HEADER,&shard.to_string())}
    /// [`Self::set_uri`], [`Self::set_shard`] and [`Self::set_read_copies`]
    /// at once: one rebuild of the metadata (a fan-out builds a shard
    /// request per shard of every search) instead of three.
    pub fn set_shard_target(&mut self,uri:&str,shard:u32,indices:&[String])->Result<()> {
        let size=indices.iter().try_fold(indices.len().saturating_sub(1),|size,value|size.checked_add(value.len())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","read copies size overflow"))?;
        let _memory=breakers().reserve_request(size as u64,"read copies header")?;
        let mut copies=String::new();copies.try_reserve_exact(size).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        for (i,index) in indices.iter().enumerate(){if i!=0{copies.push(',');}copies.push_str(index);}
        let shard=shard.to_string();
        let kept=self.headers.iter().filter(|(key,_)|!key.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)&&!key.eq_ignore_ascii_case(super::copies::READ_COPIES_HEADER)).map(|(k,v)|(k.as_str(),v.as_str()));
        let added=[(crate::shard_engines::SHARD_HEADER,shard.as_str()),(super::copies::READ_COPIES_HEADER,copies.as_str())];
        let next=self.keeping_parse(Self::from_parts(&self.method,uri,kept.chain(added),self.body.clone(),self.hops,self.dfs.clone(),self.app_plan.clone())?);
        *self=next;Ok(())
    }
    pub fn set_read_copies(&mut self,indices:&[String])->Result<()> {
        let size=indices.iter().try_fold(indices.len().saturating_sub(1),|size,value|size.checked_add(value.len())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","read copies size overflow"))?;
        let mut memory=breakers().reserve_request(size as u64,"read copies header")?;
        let mut value=String::new();value.try_reserve_exact(size).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        for (i,index) in indices.iter().enumerate(){if i!=0{value.push(',');}value.push_str(index);}
        self.replace_header(super::copies::READ_COPIES_HEADER,&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::{scoped, Breakers, Kind, RequestScope};

    fn hs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    /// One rebuild for a shard request's target gives what the three
    /// separate edits gave (headers in order, URI, shared parse).
    #[test]
    fn shard_target_is_the_three_edits_at_once() {
        let base = hs(&[("content-type", "application/json"), (crate::shard_engines::SHARD_HEADER, "0"), ("x-a", "1"), (super::super::copies::READ_COPIES_HEADER, "old")]);
        let template = RestReq::try_new("POST", "/i/_search", &base, br#"{"size":1}"#, 0, Dfs::None, None).unwrap();
        assert!(template.json_ref().is_some());
        let idx = vec!["a".to_string(), "b".to_string()];
        let mut three = template.clone();
        three.set_uri("/a,b/_search?x=1").unwrap();
        three.set_shard(3).unwrap();
        three.set_read_copies(&idx).unwrap();
        let mut one = template.clone();
        one.set_shard_target("/a,b/_search?x=1", 3, &idx).unwrap();
        assert_eq!(one.uri, three.uri);
        assert_eq!(one.headers, three.headers);
        assert!(Arc::ptr_eq(&one.parsed, &template.parsed), "the coordinator's parse is shared");
    }

    /// Header edits give the headers (order included) and the breaker
    /// charge of a request built directly with the result, and release it.
    #[test]
    fn header_edits_match_direct_construction_and_accounting() {
        let budget = Breakers::leak_for_tests(1 << 20);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let scope = RequestScope::new("header edits", budget);
        scoped(Some(scope.clone()), || {
            let base = hs(&[("Content-Type", "application/json"), ("x-a", "1"), ("X-Shard", "0"), ("x-b", "two")]);
            let mut edited = RestReq::try_new("POST", "/pmc/_search?x=1", &base, b"{}", 0, Dfs::None, None).unwrap();
            edited.replace_header("x-shard", "3").unwrap();
            edited.push_header("x-c", "c").unwrap();
            edited.retain_headers(|(k, _)| k != "x-a").unwrap();
            edited.replace_header("x-new", "n").unwrap();
            let want = hs(&[("Content-Type", "application/json"), ("x-b", "two"), ("x-shard", "3"), ("x-c", "c"), ("x-new", "n")]);
            assert_eq!(edited.headers, want);
            let used_edited = budget.get(Kind::Request).used();
            let direct = RestReq::try_new("POST", "/pmc/_search?x=1", &want, b"{}", 0, Dfs::None, None).unwrap();
            assert_eq!(budget.get(Kind::Request).used(), 2 * used_edited, "same charge for the same request");
            drop(direct);
            drop(edited);
            assert_eq!(budget.get(Kind::Request).used(), 0);
        });
        // A refused copy leaves the request as it was.
        let small = Breakers::leak_for_tests(4096);
        small.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let scope = RequestScope::new("header refusal", small);
        scoped(Some(scope.clone()), || {
            let mut r = RestReq::try_new("GET", "/", &hs(&[("a", "b")]), b"", 0, Dfs::None, None).unwrap();
            let big = "v".repeat(8192);
            assert_eq!(r.push_header("x", &big).unwrap_err().status, 429);
            assert_eq!(r.headers, hs(&[("a", "b")]));
        });
    }
}
