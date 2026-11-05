//! Allocation-bounded response media negotiation for the cluster HTTP edge.
//! Returned text is owned by the caller's retained reservation.
use super::*;
#[path="media_unicode.rs"] mod unicode;
use elasticsearch_engine::memory::Reservation;
fn denied()->EsError{EsError::new(429,"circuit_breaking_exception","media negotiation allocation refused")}
fn text(memory:&mut Reservation,args:std::fmt::Arguments<'_>)->Result<String,EsError>{
    struct Count(usize);impl std::fmt::Write for Count{fn write_str(&mut self,s:&str)->std::fmt::Result{self.0=self.0.checked_add(s.len()).ok_or(std::fmt::Error)?;Ok(())}}
    let mut count=Count(0);std::fmt::write(&mut count,args).map_err(|_|denied())?;
    memory.grow(count.0 as u64,"media negotiation text")?;
    let mut out=String::new();out.try_reserve_exact(count.0).map_err(|_|denied())?;
    struct Fixed<'a>(&'a mut String,usize);impl std::fmt::Write for Fixed<'_>{fn write_str(&mut self,s:&str)->std::fmt::Result{if s.len()>self.1.saturating_sub(self.0.len()){return Err(std::fmt::Error)}self.0.push_str(s);Ok(())}}
    std::fmt::write(&mut Fixed(&mut out,count.0),args).map_err(|_|denied())?;Ok(out)
}
struct Lower<'a>(&'a [u8]);impl std::fmt::Display for Lower<'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
    for chunk in self.0.utf8_chunks(){let value=chunk.valid();for (at,ch) in value.char_indices(){
        if ch=='Σ'{
            let before=value[..at].chars().rev().find(|c|!unicode::ignorable(*c)).is_some_and(unicode::cased);
            let after=value[at+ch.len_utf8()..].chars().find(|c|!unicode::ignorable(*c)).is_some_and(unicode::cased);
            f.write_str(if before&&!after{"ς"}else{"σ"})?;
        }else{for ch in ch.to_lowercase(){write!(f,"{ch}")?;}}
    }if !chunk.invalid().is_empty(){f.write_str("\u{fffd}")?;}}Ok(())
}}
struct Parsed<'a>{kind:&'a str,sub:&'a str,raw:&'a str}
impl<'a> Parsed<'a>{
    fn params(&self)->impl Iterator<Item=(&'a str,&'a str)> {self.raw.split(';').skip(1).filter_map(|p|{let p=p.trim();if p.is_empty(){None}else{p.split_once('=').map(|(k,v)|(k.trim(),v.trim()))}})}
    fn param(&self,key:&str)->Option<&'a str>{self.params().find(|(k,_)|*k==key).map(|(_,v)|v)}
    fn wire(&self)->Option<(&'static str,Option<wire::Wire>)>{
        if self.kind!="application"{return None}
        match self.sub {
            "json"|"x-ndjson"|"*"=>Some(("application/json",None)),
            "vnd.elasticsearch+json"|"vnd.elasticsearch+x-ndjson"=>Some(("application/vnd.elasticsearch+json",None)),
            "yaml"=>Some(("application/yaml",Some(wire::Wire::Yaml))),"vnd.elasticsearch+yaml"=>Some(("application/vnd.elasticsearch+yaml",Some(wire::Wire::Yaml))),
            "smile"=>Some(("application/smile",Some(wire::Wire::Smile))),"vnd.elasticsearch+smile"=>Some(("application/vnd.elasticsearch+smile",Some(wire::Wire::Smile))),
            "cbor"=>Some(("application/cbor",Some(wire::Wire::Cbor))),"vnd.elasticsearch+cbor"=>Some(("application/vnd.elasticsearch+cbor",Some(wire::Wire::Cbor))),_=>None
        }
    }
}
fn parse(raw:&str)->Option<Option<Parsed<'_>>>{
    if raw.contains(',')||raw=="*/*"{return Some(None)}
    let first=raw.split(';').next()?;let (kind,sub)=first.split_once('/')?;
    if sub.contains('/')||!is_token(kind.trim())||!is_token(sub.trim()){return None}
    for p in raw.split(';').skip(1){let p=p.trim();if p.is_empty(){continue}let(k,v)=p.split_once('=')?;
        if v.contains('=')||k.is_empty()||v.is_empty()||k.ends_with(char::is_whitespace)||v.starts_with(char::is_whitespace){return None}}
    Some(Some(Parsed{kind:kind.trim(),sub:sub.trim(),raw}))
}
pub(super) fn select(headers:&HeaderMap,format:Option<&str>,has_content:bool,memory:&mut Reservation)->Result<Option<(wire::Wire,String)>,EsError>{
    let accept=headers.get("accept").map(|h|text(memory,format_args!("{}",Lower(h.as_bytes())))).transpose()?;
    let content=headers.get("content-type").map(|h|text(memory,format_args!("{}",Lower(h.as_bytes())))).transpose()?;
    let Some(accept)=accept.as_deref().map(parse).unwrap_or(Some(None))else{return Ok(None)};
    let Some(content)=content.as_deref().map(parse).unwrap_or(Some(None))else{return Ok(None)};
    let version=|p:&Option<Parsed<'_>>|->Option<Option<i16>>{match p.as_ref().and_then(|p|p.param("compatible-with")){None=>Some(None),Some(v)=>v.parse::<i8>().ok().map(|v|Some(v as i16))}};
    let (Some(av),Some(cv))=(version(&accept),version(&content))else{return Ok(None)};
    let current=RestApiVersion::CURRENT.major()as i16;let minimum=RestApiVersion::MINIMUM_SUPPORTED.major()as i16;
    let(a,c)=(av.unwrap_or(current),cv.unwrap_or(current));
    if !(minimum..=current).contains(&a)||has_content&&(!(minimum..=current).contains(&c)||a!=c||av.is_none()!=cv.is_none()){return Ok(None)}
    let selected=match format{Some("json")=>Some(("application/json",None)),Some("yaml")=>Some(("application/yaml",Some(wire::Wire::Yaml))),Some("smile")=>Some(("application/smile",Some(wire::Wire::Smile))),Some("cbor")=>Some(("application/cbor",Some(wire::Wire::Cbor))),_=>None}
        .or_else(||accept.as_ref().and_then(Parsed::wire)).or_else(||content.as_ref().and_then(Parsed::wire));
    let Some((base,Some(wire)))=selected else{return Ok(None)};
    struct Mime<'a,'b>(&'static str,Option<&'a Parsed<'b>>);impl std::fmt::Display for Mime<'_,'_>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.write_str(self.0)?;if let Some(p)=self.1{for(k,v)in p.params(){write!(f,";{k}={v}")?;}}Ok(())}}
    Ok(Some((wire,text(memory,format_args!("{}",Mime(base,accept.as_ref())))?)))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn admitted_selection_matches_existing_header_contract(){
        for accept in ["application/yaml;name=ΟΣ", "application/yaml;name=ΟΣ'", "application/yaml;name=ΟΣΑ", "application/yaml;name=İ", "application/yaml;name=Σ", "application/yaml;name=ΟΣ\u{301}", "application/yaml;compatible-with=8", "text/plain", "*/*", "application/yaml;bad =x", "application/yaml;name=a=b"]{
            let mut headers=HeaderMap::new();headers.insert("accept",axum::http::HeaderValue::from_str(accept).unwrap());
            let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"media fixture").unwrap();
            assert_eq!(select(&headers,None,false,&mut memory).unwrap(),negotiate_wire(&headers,None,false),"{accept}");
        }
    }
    #[test]
    fn contextual_lowercase_preserves_sigma_and_lossy_boundaries(){
        for text in ["ΟΣ", "ΟΣΑ", "AΣ'", "AΣ'A", "AΣ\u{301}", "AΣ\u{301}A", "Σ", "İΣ", "AΣ--A"]{
            assert_eq!(format!("{}",Lower(text.as_bytes())),text.to_lowercase());
        }
    }
}
