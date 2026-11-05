//! Owned URI metadata at the standalone HTTP boundary. Each producer admits
//! its concrete capacity before allocating; the request carries its owner.
//! URI/protocol metadata belongs to native Accounting, not Elasticsearch's
//! query Request breaker. A stats request must not report its own URI storage
//! as unreleased query work. Native backing remains finitely admitted; the
//! request policy applies the RSS parent at REST in-flight/query admission.
use super::*;
use elasticsearch_engine::memory::Reservation;
pub(super) struct Builder { pub memory: Reservation }
impl Builder {
    pub fn new(_exempt: bool) -> Result<Self, EsError> {
        let breakers=elasticsearch_engine::memory::breakers();
        Ok(Self {memory:breakers.reserve_native(128,"HTTP URI owner")?})
    }
    fn charge(&mut self, bytes: usize) -> Result<(),EsError> {
        self.memory.grow(bytes as u64,"HTTP URI metadata")
    }
    pub fn text(&mut self, value:&str)->Result<String,EsError> {
        self.charge(value.len())?;
        let mut out=String::new(); out.try_reserve_exact(value.len()).map_err(|_|denied())?;
        out.push_str(value); Ok(out)
    }
    pub fn lossy(&mut self, bytes:&[u8])->Result<String,EsError> {
        let size=bytes.utf8_chunks().try_fold(0usize,|n,c|n.checked_add(c.valid().len()).and_then(|n|n.checked_add(if c.invalid().is_empty(){0}else{3}))).ok_or_else(denied)?;
        self.charge(size)?;
        let mut out=String::new();out.try_reserve_exact(size).map_err(|_|denied())?;
        for c in bytes.utf8_chunks(){out.push_str(c.valid());if !c.invalid().is_empty(){out.push('\u{fffd}')}}
        Ok(out)
    }
    pub fn decoded(&mut self, value:&str, plus:bool)->Result<String,EsError> {
        // Decoding cannot exceed input bytes. Invalid UTF-8 expansion is
        // counted separately before constructing the final String.
        self.charge(value.len())?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(value.len()).map_err(|_|denied())?;
        let source=value.as_bytes();let mut i=0;
        while i<source.len(){
            if source[i]==b'%' && i+2<source.len(){if let Some(v)=hex_pair(source[i+1],source[i+2]){bytes.push(v);i+=3;continue}}
            bytes.push(if plus&&source[i]==b'+'{b' '}else{source[i]});i+=1;
        }
        self.lossy(&bytes)
    }
    pub fn parameters(&mut self, query:Option<&str>)->Result<HashMap<String,String>,EsError>{
        let mut out=HashMap::new();
        if let Some(query)=query {for pair in query.split('&').filter(|v|!v.is_empty()) {
            let(k,v)=pair.split_once('=').unwrap_or((pair,""));
            let key=self.decoded(k,true)?;let value=self.decoded(v,true)?;
            if !out.contains_key(&key)&&out.len()==out.capacity(){
                // Hashbrown uses power-of-two buckets and 7/8 load; include
                // control bytes, allocation alignment and old/new overlap.
                let entries=out.len().checked_add(1).ok_or_else(denied)?;
                let buckets=entries.checked_mul(8).and_then(|n|n.checked_div(7)).and_then(|n|n.checked_add(1)).and_then(usize::checked_next_power_of_two).ok_or_else(denied)?.max(4);
                let bytes=buckets.checked_mul(std::mem::size_of::<(String,String)>()+1).and_then(|n|n.checked_add(64)).ok_or_else(denied)?;
                self.charge(bytes)?;out.try_reserve(1).map_err(|_|denied())?;
            }
            out.insert(key,value);
        }}
        Ok(out)
    }
    pub fn segments(&mut self,path:&str)->Result<Vec<String>,EsError>{
        let count=path.split('/').filter(|s|!s.is_empty()).count();
        self.charge(count.checked_mul(std::mem::size_of::<String>()).ok_or_else(denied)?)?;
        let mut out=Vec::new();out.try_reserve_exact(count).map_err(|_|denied())?;
        for value in path.split('/').filter(|s|!s.is_empty()){out.push(self.decoded(value,false)?)}
        Ok(out)
    }
}
fn denied()->EsError{EsError::new(429,"circuit_breaking_exception","HTTP URI metadata allocation refused")}
#[derive(Clone)]
pub(super) struct Owner(pub std::sync::Arc<std::sync::Mutex<Reservation>>);

pub(super) fn refusal(head:bool)->Response<Body>{
    let body=if head{Body::empty()}else{Body::from(Bytes::from_static(br#"{"error":{"root_cause":[{"type":"circuit_breaking_exception","reason":"HTTP URI allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"}],"type":"circuit_breaking_exception","reason":"HTTP URI allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"},"status":429}"#))};
    Response::builder().status(429).header("x-elastic-product","Elasticsearch").header("content-type","application/json").body(body).unwrap()
}

impl Builder {
    /// Plan the legacy media validator's concrete copies before entering it.
    /// This is separate from URI storage: it covers two header decodes, their
    /// lowercase buffers, split vectors, copied parameter pairs, and retained
    /// diagnostics/output MIME. It is not an estimate for application bodies.
    pub fn media(&mut self,headers:&HeaderMap)->Result<(),EsError>{
        fn add(total:&mut usize,bytes:usize)->Result<(),EsError>{*total=total.checked_add(bytes).ok_or_else(denied)?;Ok(())}
        fn buffers(length:usize)->Result<usize,EsError>{length.max(8).checked_next_power_of_two().and_then(|n|n.checked_mul(2)).ok_or_else(denied)}
        fn slots<T>(count:usize)->Result<usize,EsError>{if count==0{return Ok(0)}buffers(count.max(4))?.checked_mul(std::mem::size_of::<T>()).ok_or_else(denied)}
        fn lower_length(raw:&[u8])->Result<usize,EsError>{raw.utf8_chunks().try_fold(0usize,|mut total,chunk|{for ch in chunk.valid().chars(){for lower in ch.to_lowercase(){add(&mut total,lower.len_utf8())?;}}if !chunk.invalid().is_empty(){add(&mut total,3)?;}Ok(total)})}
        let mut total=0usize;
        for name in ["accept","content-type"]{if let Some(header)=headers.get(name){
            let bytes=header.as_bytes();let decoded=bytes.utf8_chunks().try_fold(0usize,|mut total,chunk|{add(&mut total,chunk.valid().len())?;if !chunk.invalid().is_empty(){add(&mut total,3)?;}Ok::<_,EsError>(total)})?;
            let lower=lower_length(bytes)?;
            add(&mut total,buffers(decoded.max(bytes.len()))?)?; // lossy decoder old/new overlap
            add(&mut total,buffers(lower.max(decoded))?)?; // lowercase starts with input-sized capacity
            add(&mut total,decoded)?; // ParsedMediaType.original
            let elements=bytes.split(|b|*b==b';').count();add(&mut total,slots::<&str>(elements)?)?;
            let mut parts=bytes.split(|b|*b==b';');
            let first=parts.next().unwrap_or_default();add(&mut total,slots::<&str>(first.split(|b|*b==b'/').count())?)?;
            add(&mut total,buffers(lower_length(first)?.checked_add(32).ok_or_else(denied)?)?)?; // MIME formatting
            add(&mut total,slots::<(String,String)>(elements.saturating_sub(1))?)?;
            for part in parts {
                add(&mut total,slots::<&str>(part.split(|b|*b==b'=').count())?)?;
                add(&mut total,lower_length(part)?)?; // copied parameter key + value
            }
            // Version diagnostics may display both original headers, then
            // retain the reason in EsError; the intermediate format buffer
            // overlaps the copied display strings. Header text occurs once
            // in each of these independently bounded allocations.
            add(&mut total,decoded)?;
            add(&mut total,buffers(decoded.checked_add(512).ok_or_else(denied)?)?)?;
            add(&mut total,buffers(lower.checked_add(128).ok_or_else(denied)?)?)?; // selected content type growth
        }}
        // Fixed error-kind strings, descriptor vectors, and static MIME text.
        add(&mut total,4096)?;
        self.charge(total)
    }
}

impl Builder{
    pub fn route_warning(&mut self,method:&Method,path:&str)->Result<(),EsError>{
        let length=match path{
            "/_template/*" if matches!(*method,Method::PUT|Method::POST)=>"Legacy index templates are deprecated in favor of composable templates.".len(),
            "/_monitoring/bulk"=>"Legacy monitoring API _monitoring/bulk is deprecated for removal in Elasticsearch 10.0.".len(),
            _=>return Ok(()),
        };
        self.charge(length+std::mem::size_of::<String>())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_uri_metadata_matches_decoding_and_last_parameter_wins(){
        let mut builder=Builder::new(false).unwrap();
        for value in ["a+b", "%2B+a", "%ff%fe", "%λ", "%e2%82%ac", "plain", "a%00b"]{
            assert_eq!(builder.decoded(value,true).unwrap(),decode_component(value));
            assert_eq!(builder.decoded(value,false).unwrap(),decode_path_segment(value));
        }
        let parameters=builder.parameters(Some("a=first&a=second&x=%ff&empty")).unwrap();
        assert_eq!(parameters["a"],"second");assert_eq!(parameters["x"],"\u{fffd}");assert_eq!(parameters["empty"],"");
        assert_eq!(builder.segments("/a%2Fb/+/%λ/").unwrap(),vec!["a/b","+","%λ"]);
    }
    #[test]
    fn denied_metadata_does_not_allocate_or_drop_a_live_input(){
        let budget=elasticsearch_engine::memory::Breakers::leak_for_tests(64);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let mut builder=Builder{memory:budget.reserve_request(0,"URI fixture").unwrap()};
        assert_eq!(builder.parameters(Some("a=b")).unwrap_err().status,429);
        drop(builder);assert_eq!(budget.request.reserved(),0);
    }
}
impl Owner {
    pub fn bytes(&self,value:&[u8],_exempt:bool)->Result<Bytes,EsError>{
        struct Payload{bytes:Vec<u8>,_owner:Owner}impl AsRef<[u8]> for Payload{fn as_ref(&self)->&[u8]{&self.bytes}}
        let size=value.len().checked_add(std::mem::size_of::<Payload>()+4*std::mem::size_of::<usize>()).ok_or_else(denied)?;
        {let mut memory=self.0.lock().unwrap();memory.grow(size as u64,"HTTP source position copy")?}
        let mut bytes=Vec::new();bytes.try_reserve_exact(value.len()).map_err(|_|denied())?;bytes.extend_from_slice(value);
        Ok(Bytes::from_owner(Payload{bytes,_owner:self.clone()}))
    }
}

impl Owner {
    pub fn formatted(&self,value:impl std::fmt::Display,_exempt:bool)->Result<String,EsError>{
        struct Count(usize);impl std::fmt::Write for Count{fn write_str(&mut self,value:&str)->std::fmt::Result{self.0=self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;Ok(())}}
        let mut count=Count(0);std::fmt::write(&mut count,format_args!("{value}")).map_err(|_|denied())?;
        {let mut memory=self.0.lock().unwrap();memory.grow(count.0 as u64,"HTTP request diagnostic metadata")?}
        let mut out=String::new();out.try_reserve_exact(count.0).map_err(|_|denied())?;
        struct Fixed<'a>(&'a mut String,usize);impl std::fmt::Write for Fixed<'_>{fn write_str(&mut self,value:&str)->std::fmt::Result{if value.len()>self.1.saturating_sub(self.0.len()){return Err(std::fmt::Error)}self.0.push_str(value);Ok(())}}
        std::fmt::write(&mut Fixed(&mut out,count.0),format_args!("{value}")).map_err(|_|denied())?;Ok(out)
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn detached_scope_retains_its_counted_uri_label_owner(){
        use elasticsearch_engine::memory::{Breakers,RequestScope};
        let breakers=Breakers::leak_for_tests(1<<20);
        breakers.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let owner=Owner(std::sync::Arc::new(std::sync::Mutex::new(breakers.reserve_request(0,"diagnostic fixture").unwrap().into_result_owner())));
        let label=owner.formatted(format_args!("GET {}","/an-index/_search"),false).unwrap();
        assert_eq!(label,"GET /an-index/_search");
        let scope=RequestScope::with_label_owner(label,breakers,owner.0.clone());
        let worker=scope.clone();drop(owner);drop(scope);
        assert!(breakers.request.reserved()>0);
        drop(worker);assert_eq!(breakers.request.reserved(),0);
    }
}
