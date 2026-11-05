//! Finalized outgoing HTTP metadata. Text ownership follows individual names
//! and values; internal map storage ownership follows consuming iterators.
//! HTTP encoders may retain empty map capacity between responses. Keep this
//! protocol storage in Accounting with its allocation owner, rather than
//! presenting it as unfinished query work in the Request breaker. Parent
//! admission still applies before every allocation.
//! These maps must not be deep-cloned or mutated via infallible generic APIs.
use axum::{body::Body,http::{HeaderMap,HeaderName,HeaderValue,Response,StatusCode}};
use bytes::Bytes;
use elasticsearch_engine::{error::{EsError,Result},memory::{self,Reservation}};
/// One name's or value's bytes. The texts of one map share a reservation
/// (one breaker round trip per map, not two per header), released once
/// the last of them is dropped.
struct Text { bytes:Vec<u8>, _owner:std::sync::Arc<Reservation> }
impl AsRef<[u8]> for Text {fn as_ref(&self)->&[u8]{&self.bytes}}
/// The admitted size of a header text.
fn text_bytes(value:&str)->u64 {(value.len() as u64).saturating_add(128)}
fn text(value:&str, lowercase:bool, owner:&std::sync::Arc<Reservation>)->Result<Bytes> {
    let mut bytes=Vec::new();bytes.try_reserve_exact(value.len()).map_err(|_|refused())?;
    bytes.extend(value.bytes().map(|b|if lowercase{b.to_ascii_lowercase()}else{b}));
    Ok(Bytes::from_owner(Text{bytes,_owner:owner.clone()}))
}
fn refused()->EsError {EsError::new(429,"circuit_breaking_exception","outgoing header allocation refused")}
/// No temporary Vec of header pairs: the cloned iterator only borrows input.
pub fn collect<'a,I>(headers:I)->Result<HeaderMap>
where I:Iterator<Item=(&'a str,&'a str)>+Clone { collect_inner(headers,false) }
/// Internal HTTP adapters historically ignore malformed textual headers; keep
/// that behavior while allocation refusals remain errors.
pub fn collect_valid<'a,I>(headers:I)->Result<HeaderMap>
where I:Iterator<Item=(&'a str,&'a str)>+Clone { collect_inner(headers,true) }
fn collect_inner<'a,I>(headers:I, ignore_invalid:bool)->Result<HeaderMap>
where I:Iterator<Item=(&'a str,&'a str)>+Clone {
    let (count,bytes)=headers.clone().fold((0usize,0u64),|(n,b),(name,value)|(n+1,b.saturating_add(text_bytes(name)).saturating_add(text_bytes(value))));
    collect_counted(headers,count,bytes,ignore_invalid)
}
/// `bytes`: the admitted size of every name and value of `headers` (the
/// sum of their [`text_bytes`]), reserved at once.
fn collect_counted<'a,I>(mut headers:I,count:usize,bytes:u64,ignore_invalid:bool)->Result<HeaderMap>
where I:Iterator<Item=(&'a str,&'a str)> {
    let owner=std::sync::Arc::new(memory::breakers().reserve_native(bytes,"outgoing header text")?.into_result_owner());
    let failure=std::cell::RefCell::new(None);
    let items=(0..count).map(|_| {
        if failure.borrow().is_some(){return None;}
        let Some((name,value))=headers.next() else {
            *failure.borrow_mut()=Some(EsError::internal("header iterator changed between passes"));
            return None;
        };
        let built=(||{
            let name=HeaderName::from_shared_lowercase(text(name,true,&owner)?).map_err(|_|EsError::internal("invalid outgoing header name"))?;
            let value=HeaderValue::from_maybe_shared(text(value,false,&owner)?).map_err(|_|EsError::internal("invalid outgoing header value"))?;
            Ok::<_,EsError>((name,value))
        })();
        match built {Ok(value)=>Some(value),Err(error) if ignore_invalid && error.status==500 => None,Err(error)=>{*failure.borrow_mut()=Some(error);None}}
    });
    let map=HeaderMap::try_from_admitted_iter(items,|bytes| {
        match memory::breakers().reserve_native(bytes as u64,"outgoing header map") {
            Ok(owner)=>Ok(axum::http::header::HeaderStorageOwner::new(owner.into_result_owner())),
            Err(error)=>{*failure.borrow_mut()=Some(error);Err(axum::http::header::MaxSizeReached::admission_refused())}
        }
    });
    if let Some(error)=failure.into_inner(){return Err(error);}
    let map=map.map_err(|_|refused())?;
    if headers.next().is_some(){return Err(EsError::internal("header iterator changed between passes"));}
    Ok(map)
}
/// Rebuild finalized headers with a content length without cloning a map
/// iterator or allocating a temporary list/decimal string before admission.
pub fn with_content_length(headers:&HeaderMap,length:u64)->Result<HeaderMap> {
    let mut digits=[0u8;20];let mut offset=digits.len();let mut remaining=length;
    loop {offset-=1;digits[offset]=b'0'+(remaining%10) as u8;remaining/=10;if remaining==0{break;}}
    let length=std::str::from_utf8(&digits[offset..]).expect("decimal digits are UTF-8");
    // HeaderMap::Iter is not Clone. Recreate the same immutable borrowed view
    // for the sizing and encoding passes; every backing allocation is admitted.
    let make=||headers.iter().filter_map(|(name,value)|value.to_str().ok().map(|value|(name.as_str(),value)))
        .chain(std::iter::once(("content-length",length)));
    let (count,bytes)=make().fold((0usize,0u64),|(n,b),(name,value)|(n+1,b.saturating_add(text_bytes(name)).saturating_add(text_bytes(value))));
    collect_counted(make(),count,bytes,false)
}

pub fn response<'a,I>(status:u16,headers:I,body:Body)->Result<Response<Body>>
where I:Iterator<Item=(&'a str,&'a str)>+Clone {
    let headers=collect(headers)?;
    let mut response=Response::new(body);
    *response.status_mut()=StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    *response.headers_mut()=headers;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_capacity_stays_accounted_until_transport_releases_it() {
        let breakers=memory::Breakers::leak_for_tests(1 << 20);
        let scope=memory::RequestScope::new("header ownership",breakers);
        let mut headers=memory::scoped(Some(scope),||collect(std::iter::once(("warning","retained")))).unwrap();
        assert_eq!(breakers.request.reserved(),0);
        assert!(breakers.accounting.reserved()>0);
        headers.clear();
        // Clearing values does not free the map's backing capacity.
        assert!(breakers.accounting.reserved()>0);
        drop(headers);
        assert_eq!(breakers.accounting.reserved(),0);
    }
    #[test]
    fn borrowed_map_reconstruction_keeps_duplicates_and_exact_decimal() {
        let mut source=HeaderMap::new();source.append("warning",HeaderValue::from_static("first"));source.append("warning",HeaderValue::from_static("second"));
        for length in [0,1,u64::MAX] {
            let copied=with_content_length(&source,length).unwrap();
            assert_eq!(copied.get_all("warning").iter().map(|v|v.to_str().unwrap()).collect::<Vec<_>>(),vec!["first","second"]);
            assert_eq!(copied["content-length"].to_str().unwrap(),length.to_string());
            assert!(!source.contains_key("content-length"));
        }
    }
    #[test]
    fn counted_iterator_change_is_an_error_not_growth_or_panic() {
        assert!(collect_counted(std::iter::empty(),1,0,false).is_err());
        assert!(collect_counted(std::iter::once(("warning","extra")),0,0,false).is_err());
        // Capacity refusal occurs before consuming the iterator and must not
        // be mislabeled as a changed second pass. No huge allocation occurs.
        let error=collect_counted(std::iter::once(("warning","extra")),usize::MAX,0,false).unwrap_err();
        assert_eq!(error.status,429);
    }
}
