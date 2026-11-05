//! Immutable, shared REST response payloads. Copies share admitted bytes and
//! headers; edits construct an independently admitted replacement before swap.
use super::admitted::{bytes, slots, text};
use axum::body::Bytes;
use elasticsearch_engine::{error::EsError, memory::{breakers, Reservation}};
use serde::{Serialize, Serializer};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug)]
pub struct RestResp { pub status: u16, payload: Arc<Payload> }
pub struct Payload {
    pub headers: Vec<(String,String)>,
    pub body: Bytes,
    refused: bool,
    // Header allocations are owned independently of body extraction.
    _memory: Option<Reservation>,
}
impl std::fmt::Debug for Payload {
    fn fmt(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.debug_struct("Payload").field("headers",&self.headers).field("body",&self.body).finish()}
}
impl std::ops::Deref for RestResp { type Target=Payload; fn deref(&self)->&Payload{&self.payload} }
struct BodyOwner { value:Vec<u8>, _memory:Reservation }
impl AsRef<[u8]> for BodyOwner {fn as_ref(&self)->&[u8]{&self.value}}
fn allocation(e:impl std::fmt::Display)->EsError{EsError::new(429,"circuit_breaking_exception",e.to_string())}
fn owner_bytes()->usize{std::mem::size_of::<BodyOwner>()+4*std::mem::size_of::<usize>()}
impl RestResp {
    pub(super) fn decoded(status:u16,headers:Vec<(String,String)>,mut header_memory:Reservation,body:Vec<u8>,mut body_memory:Reservation)->Result<Self,EsError>{
        header_memory.grow((std::mem::size_of::<Payload>()+2*std::mem::size_of::<usize>()) as u64,"REST response header owner")?;
        body_memory.grow(owner_bytes() as u64,"REST response body owner")?;
        let body=Bytes::from_owner(BodyOwner{value:body,_memory:body_memory.into_result_owner()});
        Ok(Self{status,payload:Arc::new(Payload{headers,body,refused:false,_memory:Some(header_memory.into_result_owner())})})
    }
    /// A fixed, process-owned emergency response is required when even error
    /// rendering is refused. It cannot recurse into the exhausted allocator.
    fn refused()->Self {
        static VALUE:OnceLock<Arc<Payload>>=OnceLock::new();
        Self{status:429,payload:VALUE.get_or_init(||Arc::new(Payload{
            headers:vec![("content-type".into(),"application/json".into())],refused:true,body:Bytes::from_static(br#"{"error":{"root_cause":[{"type":"circuit_breaking_exception","reason":"REST response allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"}],"type":"circuit_breaking_exception","reason":"REST response allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"},"status":429}"#),_memory:None,
        })).clone()}
    }
    /// Only an already-decided authentication rejection may use this fixed
    /// process-owned response if rendering its details is refused.
    fn unauthorized(head:bool)->Self {
        static VALUE:OnceLock<Arc<Payload>>=OnceLock::new();
        static HEAD:OnceLock<Arc<Payload>>=OnceLock::new();
        Self{status:401,payload:(if head{&HEAD}else{&VALUE}).get_or_init(||Arc::new(Payload{
            headers:vec![("content-type".into(),"application/json".into()),("www-authenticate".into(),"Basic realm=\"security\", charset=\"UTF-8\"".into())],
            refused:true,body:if head{Bytes::new()}else{Bytes::from_static(br#"{"error":{"type":"security_exception","reason":"authentication required"},"status":401}"#)},_memory:None,
        })).clone()}
    }
    pub(super) fn fallback(status:u16,head:bool)->Self {if status==401{Self::unauthorized(head)}else{Self::refused()}}
    pub(super) fn http_metadata(response:&axum::http::Response<axum::body::Body>,head:bool)->Result<Self,EsError> {
        let status=response.status().as_u16();
        Self::with_payload(status,response.headers().iter().filter_map(|(k,v)|Some((k.as_str(),v.to_str().ok()?))),Bytes::new())
            .or_else(|error|if status==401{Ok(Self::unauthorized(head))}else{Err(error)})
    }
    fn headers_from<'a>(items:impl Iterator<Item=(&'a str,&'a str)>, memory:&mut Reservation)->Result<Vec<(String,String)>,EsError>{
        let mut out=Vec::new();
        for (name,value) in items {
            if out.len()==out.capacity(){
                let old=out.capacity();let count=old.checked_mul(2).unwrap_or(usize::MAX).max(1);
                let size=count.checked_mul(std::mem::size_of::<(String,String)>()).ok_or_else(||allocation("REST headers overflow"))?;
                memory.grow(size as u64,"REST header replacement")?;
                out.try_reserve_exact(count-out.len()).map_err(allocation)?;
                memory.shrink((old*std::mem::size_of::<(String,String)>()) as u64);
            }
            out.push((text(memory,name)?,text(memory,value)?));
        }
        Ok(out)
    }
    pub(super) fn with_payload<'a>(status:u16,items:impl Iterator<Item=(&'a str,&'a str)>,body:Bytes)->Result<Self,EsError>{
        let mut memory=breakers().reserve_request(0,"REST response headers")?;
        let headers=Self::headers_from(items,&mut memory)?;
        memory.grow((std::mem::size_of::<Payload>()+2*std::mem::size_of::<usize>()) as u64,"REST response owner")?;
        Ok(Self{status,payload:Arc::new(Payload{headers,body,refused:false,_memory:Some(memory.into_result_owner())})})
    }
    fn copy_body(value:&[u8])->Result<Bytes,EsError>{
        let mut memory=breakers().reserve_request(owner_bytes() as u64,"REST response body")?;
        let value=bytes(&mut memory,value)?;
        Ok(Bytes::from_owner(BodyOwner{value,_memory:memory.into_result_owner()}))
    }
    pub(super) fn is_admission_refused(&self)->bool {self.payload.refused}
    /// Swap in a small, bounded body (`limit` bytes at most) without breaker
    /// admission, keeping the headers but `content-length`. Only for fixed
    /// rewrites whose refusal would leak internal state (an engine-local task
    /// id the cluster cannot resolve); anything larger is left as it was.
    pub(crate) fn replace_small_body_unadmitted(&mut self,body:String,limit:usize)->bool {
        if self.payload.refused||body.len()>limit {return false;}
        let headers=self.headers.iter().filter(|(k,_)|!k.eq_ignore_ascii_case("content-length")).cloned().collect();
        self.payload=Arc::new(Payload{headers,body:Bytes::from(body),refused:false,_memory:None});
        true
    }
    pub(super) async fn collect(self,body:axum::body::Body,head:bool)->Self {
        if self.payload.refused { return self; }
        use futures::StreamExt;
        let mut memory=match breakers().reserve_request(owner_bytes() as u64,"REST collected response"){Ok(m)=>m,Err(_)=>return Self::fallback(self.status,head)};
        let mut bytes=Vec::<u8>::new();let mut stream=body.into_data_stream();
        while let Some(chunk)=stream.next().await {
            let chunk=match chunk{Ok(c)=>c,Err(e)=>return if self.status==401{Self::unauthorized(head)}else{Self::error(500,"transport_exception",&e.to_string())}};
            let Some(required)=bytes.len().checked_add(chunk.len()) else{return Self::fallback(self.status,head)};
            if required>bytes.capacity(){
                let old=bytes.capacity();let capacity=old.checked_mul(2).unwrap_or(usize::MAX).max(required);
                if memory.grow(capacity as u64,"REST collected body replacement").is_err(){return Self::fallback(self.status,head);}
                if bytes.try_reserve_exact(capacity-bytes.len()).is_err(){return Self::fallback(self.status,head);}
                memory.shrink(old as u64);
            }
            bytes.extend_from_slice(&chunk);
        }
        let body=Bytes::from_owner(BodyOwner{value:bytes,_memory:memory.into_result_owner()});
        Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body).unwrap_or_else(|_|Self::fallback(self.status,head))
    }
    /// An in-process engine answer handed over typed
    /// ([`elasticsearch_http::TypedResponse`]): what [`Self::http_metadata`]
    /// and [`Self::collect`] make of the HTTP response it stands for, with
    /// its body taken over instead of copied.
    pub(super) fn typed(typed:elasticsearch_http::TypedResponse,head:bool)->Self {
        let elasticsearch_http::TypedResponse{status,headers,body}=typed;
        let memory=match breakers().reserve_request((owner_bytes() as u64).saturating_add(body.capacity() as u64),"REST collected response"){Ok(m)=>m,Err(_)=>return Self::fallback(status,head)};
        let body=Bytes::from_owner(BodyOwner{value:body.into_bytes(),_memory:memory.into_result_owner()});
        Self::with_payload(status,headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body).unwrap_or_else(|_|Self::fallback(status,head))
    }
    pub fn new(status:u16,headers:Vec<(String,String)>,body:impl AsRef<[u8]>)->Self{
        let result=Self::copy_body(body.as_ref()).and_then(|body|Self::with_payload(status,headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body));
        result.unwrap_or_else(|_|Self::fallback(status,body.as_ref().is_empty()))
    }
    fn render_json(value:&impl serde::Serialize)->Result<Bytes,EsError>{
        struct Count(usize);
        impl std::io::Write for Count{
            fn write(&mut self,b:&[u8])->std::io::Result<usize>{self.0=self.0.checked_add(b.len()).ok_or_else(||std::io::Error::other("JSON length overflow"))?;Ok(b.len())}
            fn flush(&mut self)->std::io::Result<()>{Ok(())}
        }
        let render=||->Result<Bytes,EsError>{
            let mut count=Count(0);serde_json::to_writer(&mut count,value).map_err(allocation)?;
            let mut memory=breakers().reserve_request(owner_bytes() as u64,"REST JSON body")?;
            let mut out=slots::<u8>(&mut memory,count.0)?;out.resize(count.0,0);
            serde_json::to_writer(out.as_mut_slice(),value).map_err(allocation)?;
            Ok(Bytes::from_owner(BodyOwner{value:out,_memory:memory.into_result_owner()}))
        };
        render()
    }
    pub(crate) fn from_es_error(error:&EsError)->Self {
        elasticsearch_engine::memory::native_control(|| elasticsearch_http::render_error_admitted(error)
            .and_then(|body|Self::with_payload(error.status,std::iter::once(("content-type","application/json")),body))
            .unwrap_or_else(|_|if error.status==401{Self::unauthorized(false)}else{Self::refused()}))
    }
    pub(crate) fn json_serialized(status:u16,value:&impl serde::Serialize)->Self {
        Self::render_json(value).and_then(|body|Self::with_payload(status,std::iter::once(("content-type","application/json")),body)).unwrap_or_else(|_|Self::fallback(status,false))
    }
    pub fn json(status:u16,value:&serde_json::Value)->Self {
        Self::render_json(value).and_then(|body|Self::with_payload(status,std::iter::once(("content-type","application/json")),body)).unwrap_or_else(|_|Self::fallback(status,false))
    }
    pub fn replace_json(&mut self,value:&serde_json::Value){
        if self.payload.refused{return;}
        let result=Self::render_json(value).and_then(|body|Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body));
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
    /// [`replace_json`](Self::replace_json) dropping `content-length`, but a
    /// refused rendering leaves this response as it was.
    pub fn try_replace_json(&mut self,value:&serde_json::Value)->Result<(),EsError>{
        if self.payload.refused{return Ok(());}
        let body=Self::render_json(value)?;
        let next=Self::with_payload(self.status,self.headers.iter().filter(|(k,_)|!k.eq_ignore_ascii_case("content-length")).map(|(k,v)|(k.as_str(),v.as_str())),body)?;
        *self=next;Ok(())
    }
    pub fn text(status:u16,value:String)->Self{
        Self::copy_body(value.as_bytes()).and_then(|body|Self::with_payload(status,std::iter::once(("content-type","text/plain; charset=UTF-8")),body)).unwrap_or_else(|_|Self::fallback(status,value.is_empty()))
    }
    /// [`replace_body`](Self::replace_body) adopting a rendered body instead
    /// of copying it: its capacity is admitted on the request breaker and the
    /// vector becomes the body's owner. A scroll page or a merged search
    /// response is megabytes of `_source`; the copy was pure overhead.
    pub fn replace_body_vec(&mut self,value:Vec<u8>){
        if self.payload.refused {return;}
        let empty=self.body.is_empty();
        let result=breakers().reserve_request((owner_bytes()+value.capacity()) as u64,"REST response body")
            .map(|memory|Bytes::from_owner(BodyOwner{value,_memory:memory.into_result_owner()}))
            .and_then(|body|Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body));
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,empty));
    }
    pub fn replace_body(&mut self,value:impl AsRef<[u8]>){
        if self.payload.refused {return;}
        let result=Self::copy_body(value.as_ref()).and_then(|body|Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body));
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
    /// Transform a borrowed response through a counted, fixed-capacity encoder.
    pub(crate) fn with_encoded_body(&self,encode:impl Fn(&mut dyn std::io::Write)->std::io::Result<()>)->Self {
        if self.payload.refused {return self.clone();}
        let result=super::admitted::encoded_body(encode,breakers()).and_then(|body|Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),body));
        result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()))
    }
    pub fn retain_headers(&mut self,mut keep:impl FnMut(&(String,String))->bool){
        if self.payload.refused {return;}
        let result=Self::with_payload(self.status,self.headers.iter().filter(|h|keep(h)).map(|(k,v)|(k.as_str(),v.as_str())),self.body.clone());
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
    pub(super) fn append_header(&mut self,name:&str,value:&str)->Result<(),EsError>{
        let next=Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())).chain(std::iter::once((name,value))),self.body.clone())?;
        *self=next;Ok(())
    }
    pub(super) fn append_json_header<T:serde::Serialize+?Sized>(&mut self,name:&str,value:&T)->Result<(),EsError>{
        let bytes=elasticsearch_http::wire::json_admitted(value,false)?;
        let value=std::str::from_utf8(&bytes).map_err(allocation)?;
        self.append_header(name,value)
    }
    pub fn push_header(&mut self,header:(String,String)){
        if self.payload.refused {return;}
        let result=Self::with_payload(self.status,self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())).chain(std::iter::once((header.0.as_str(),header.1.as_str()))),self.body.clone());
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
    pub fn extend_headers(&mut self,headers:impl IntoIterator<Item=(String,String)>){
        if self.payload.refused {return;}
        let build=||->Result<Self,EsError>{
            let mut memory=breakers().reserve_request(0,"REST extended headers")?;
            let mut out=Self::headers_from(self.headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),&mut memory)?;
            for (name,value) in headers {
                if out.len()==out.capacity(){
                    let old=out.capacity();let count=old.checked_mul(2).unwrap_or(usize::MAX).max(1);
                    let size=count.checked_mul(std::mem::size_of::<(String,String)>()).ok_or_else(||allocation("REST headers overflow"))?;
                    memory.grow(size as u64,"REST header replacement")?;
                    out.try_reserve_exact(count-out.len()).map_err(allocation)?;
                    memory.shrink((old*std::mem::size_of::<(String,String)>()) as u64);
                }
                out.push((text(&mut memory,&name)?,text(&mut memory,&value)?));
            }
            memory.grow((std::mem::size_of::<Payload>()+2*std::mem::size_of::<usize>()) as u64,"REST response owner")?;
            Ok(Self{status:self.status,payload:Arc::new(Payload{headers:out,body:self.body.clone(),refused:false,_memory:Some(memory.into_result_owner())})})
        };
        *self=build().unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
    pub fn replace_headers(&mut self,headers:Vec<(String,String)>){
        if self.payload.refused {return;}
        let result=Self::with_payload(self.status,headers.iter().map(|(k,v)|(k.as_str(),v.as_str())),self.body.clone());
        *self=result.unwrap_or_else(|_|Self::fallback(self.status,self.body.is_empty()));
    }
}
impl Serialize for RestResp {
    fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
        use serde::ser::SerializeStruct;
        struct Body<'a>(&'a [u8]);impl Serialize for Body<'_>{fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok,S::Error>{s.serialize_bytes(self.0)}}
        let mut out=s.serialize_struct("RestResp",3)?;out.serialize_field("status",&self.status)?;out.serialize_field("headers",&self.headers)?;out.serialize_field("body",&Body(&self.body))?;out.end()
    }
}

/// Carries newly copied HTTP header ownership through the HTTP body's lifetime.
/// The input body retains its own distinct reservation through Bytes cloning.
struct HeaderOwner {bytes:Bytes,_memory:Reservation}
impl AsRef<[u8]> for HeaderOwner {fn as_ref(&self)->&[u8]{&self.bytes}}
pub(super) fn header_owner_bytes()->usize {std::mem::size_of::<HeaderOwner>()+4*std::mem::size_of::<usize>()}
pub(super) fn with_header_owner(bytes:Bytes,memory:Reservation)->Bytes {
    Bytes::from_owner(HeaderOwner{bytes,_memory:memory.into_result_owner()})
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
    fn fixture_breakers(bytes: u64) -> &'static Breakers {
        let budget=Breakers::leak_for_tests(bytes);
        // Exercise response ownership/refusal without unrelated process RSS.
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        budget
    }
    #[test]
    fn admitted_error_preserves_details_and_body_ownership() {
        let error=EsError::new(400,"x_content_parse_exception","outer")
            .with("index","docs").caused_by(EsError::new(400,"illegal_argument_exception","x".repeat(4096)));
        let expected=error.to_json();
        let budget=fixture_breakers(1<<20);let scope=RequestScope::new("error ownership",budget);
        let response=scoped(Some(scope.clone()),||RestResp::from_es_error(&error));
        assert_eq!(response.status,400);assert_eq!(response.body_json().unwrap(),expected);
        let body=response.body.clone();drop(response);drop(scope);
        assert_eq!(budget.get(Kind::Request).used(),0);
        assert!(budget.get(Kind::Accounting).used()>0);
        drop(body);assert_eq!(budget.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn authentication_rejection_survives_render_and_header_refusal() {
        use axum::body::HttpBody;
        let error=EsError::new(401,"security_exception","x".repeat(65536));
        let budget=fixture_breakers(1);let scope=RequestScope::new("auth response refusal",budget);
        let response=scoped(Some(scope.clone()),||RestResp::from_es_error(&error));
        assert_eq!(response.status,401);
        assert_eq!(response.body_json().unwrap()["error"]["reason"],"authentication required");
        let http=scoped(Some(scope.clone()),||super::super::to_http(response));
        assert_eq!(http.status(),401);assert!(http.headers().contains_key("www-authenticate"));
        assert!(http.body().size_hint().exact().unwrap()>0);
        let head=RestResp::with_payload(401,std::iter::once(("www-authenticate","Basic realm=\"security\"")),Bytes::new()).unwrap();
        let http=scoped(Some(scope.clone()),||super::super::to_http(head));
        assert_eq!(http.status(),401);assert!(http.headers().contains_key("www-authenticate"));
        assert_eq!(http.body().size_hint().exact(),Some(0));
        drop(http);drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
    }
    #[test]
    fn authentication_decision_survives_router_collection_and_transforms() {
        use axum::body::{Body,HttpBody};
        let budget=fixture_breakers(1);let scope=RequestScope::new("router auth refusal",budget);
        for head in [false,true] {
            let response=axum::http::Response::builder().status(401).header("www-authenticate","Basic realm=\"security\"")
                .body(if head{Body::empty()}else{Body::from("denied")}).unwrap();
            let metadata=scoped(Some(scope.clone()),||RestResp::http_metadata(&response,head)).unwrap();
            assert_eq!(metadata.status,401);assert_eq!(metadata.body.is_empty(),head);
            let metadata=RestResp::with_payload(401,std::iter::once(("www-authenticate","Basic")),Bytes::new()).unwrap();
            let collected=scoped(Some(scope.clone()),||futures::executor::block_on(metadata.collect(response.into_body(),head)));
            assert_eq!(collected.status,401);assert_eq!(collected.body.is_empty(),head);
            let http=scoped(Some(scope.clone()),||super::super::to_http(collected));
            assert_eq!(http.status(),401);assert_eq!(http.body().size_hint().exact()==Some(0),head);
        }
        let original=RestResp::with_payload(401,std::iter::once(("www-authenticate","Basic")),Bytes::from_static(b"denied")).unwrap();
        let mut headers=original.clone();scoped(Some(scope.clone()),||headers.retain_headers(|_|true));
        assert_eq!(headers.status,401);
        let mut transformed=original.clone();scoped(Some(scope.clone()),||transformed.replace_json(&serde_json::json!({"error":"denied"})));
        assert_eq!(transformed.status,401);
        let encoded=scoped(Some(scope.clone()),||original.with_encoded_body(|out|out.write_all(b"denied")));
        assert_eq!(encoded.status,401);
        drop(headers);drop(transformed);drop(encoded);drop(original);drop(scope);
        assert_eq!(budget.get(Kind::Request).used(),0);
    }
    #[test]
    fn admitted_error_refusal_is_bounded_and_nonrecursive() {
        let error=EsError::new(400,"illegal_argument_exception","x".repeat(65536));
        let budget=fixture_breakers(1024);let scope=RequestScope::new("error refusal",budget);
        let response=scoped(Some(scope.clone()),||RestResp::from_es_error(&error));
        assert_eq!(response.status,429);
        assert_eq!(response.body_json().unwrap()["error"]["type"],"circuit_breaking_exception");
        drop(response);drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
    }
    #[test]
    fn response_clone_and_extracted_body_keep_independent_payload_owners() {
        let budget=fixture_breakers(1<<20);
        let scope=RequestScope::new("response ownership",budget);
        let mut original=scoped(Some(scope.clone()),||RestResp::json(200,&serde_json::json!({"v":"x".repeat(4096)})));
        assert_eq!(original.status,200);
        let charged=budget.get(Kind::Request).used();
        let copy=original.clone();let extracted=original.body.clone();
        assert_eq!(budget.get(Kind::Request).used(),charged,"clones share actual payload, not uncharged deep copies");
        scoped(Some(scope.clone()),||original.replace_body(b"changed"));
        assert_eq!(&*original.body,b"changed");
        assert_eq!(copy.body,extracted);
        drop(original);drop(copy);drop(scope);
        assert!(budget.get(Kind::Request).used()>0,"extracted body survives executor scope");
        drop(extracted);assert_eq!(budget.get(Kind::Request).used(),0);
    }
    #[test]
    fn response_wire_matches_legacy_and_decoded_clone_survives_source() {
        #[derive(serde::Serialize)] struct Legacy {status:u16,headers:Vec<(String,String)>,body:Vec<u8>}
        let old=Legacy{status:201,headers:vec![("warning".into(),"a".repeat(1024))],body:vec![42;4096]};
        let wire=bincode::serialize(&old).unwrap();
        let budget=fixture_breakers(1<<20);let scope=RequestScope::new("response wire",budget);
        let decoded:RestResp=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();
        assert_eq!(bincode::serialize(&decoded).unwrap(),wire);
        let clone=decoded.clone();drop(decoded);drop(wire);drop(old);drop(scope);
        assert!(budget.get(Kind::Request).used()>=4096,"successful body retains request admission");
        assert!(budget.get(Kind::Accounting).used()>=1024,"protocol headers retain native admission");
        assert_eq!(clone.headers[0].1.len(),1024);
        drop(clone);assert_eq!(budget.get(Kind::Request).used(),0);
        assert_eq!(budget.get(Kind::Accounting).used(),0);
    }
    #[test]
    fn response_mutation_refusal_is_error_and_does_not_corrupt_shared_copy() {
        let budget=fixture_breakers(4096);let scope=RequestScope::new("response refusal",budget);
        let mut response=scoped(Some(scope.clone()),||RestResp::json(200,&serde_json::json!({"ok":true})));
        let original=response.clone();
        scoped(Some(scope.clone()),||response.replace_body(vec![0;65536]));
        assert_eq!(response.status,429);assert_eq!(original.status,200);
        assert_eq!(original.body_json().unwrap()["ok"],true);
        response.push_header(("unused".into(),"value".into()));
        assert_eq!(response.status,429,"following edits cannot turn refusal into success");
        drop(response);drop(original);drop(scope);assert_eq!(budget.get(Kind::Request).used(),0);
    }

    /// An adopted body is the same response as a copied one (body, status,
    /// headers), and its admission is released with the response.
    #[test]
    fn adopted_body_matches_copied_body_and_releases() {
        use elasticsearch_engine::memory::{scoped,Breakers,Kind,RequestScope};
        let budget=Breakers::leak_for_tests(64<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("adopted body",budget);
        scoped(Some(scope.clone()),||{
            let body=format!(r#"{{"hits":{{"hits":["{}"]}}}}"#,"x".repeat(1<<20)).into_bytes();
            let mut copied=RestResp::json(200,&serde_json::Value::Null);copied.replace_body(body.clone());
            let mut adopted=RestResp::json(200,&serde_json::Value::Null);adopted.replace_body_vec(body.clone());
            assert_eq!((adopted.status,&adopted.headers,&adopted.body),(copied.status,&copied.headers,&copied.body));
            drop(copied);
            assert!(budget.get(Kind::Request).used()>=body.len() as u64);
            drop(adopted);
            assert_eq!(budget.get(Kind::Request).used(),0);
        });
    }
}
