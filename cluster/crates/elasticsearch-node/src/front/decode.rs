//! Seeded forwarding wire fields. Each allocation belongs to the eventual
//! request owner; no decoded Vec/String is allocated before admission.
use super::{AppSearchPlan, Dfs, RestReq,SharedDfsStats};
use super::request_shared::PlanData;
use elasticsearch_engine::memory::Reservation;
use elasticsearch_http::{DfsField, DfsStats, DfsTerm};
use serde::de::{self, DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use std::marker::PhantomData;

trait Owned: Sized {
    fn decode<'de, D: Deserializer<'de>>(memory: &mut Reservation, d: D) -> Result<Self, D::Error>;
}
struct Seed<'a, T>(&'a mut Reservation, PhantomData<T>);
fn seed<T>(memory: &mut Reservation) -> Seed<'_, T> { Seed(memory, PhantomData) }
impl<'de, T: Owned> DeserializeSeed<'de> for Seed<'_, T> {
    type Value = T;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<T, D::Error> { T::decode(self.0, d) }
}
impl Owned for String {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error> {
        crate::replication::decode::Text(memory).deserialize(d)
    }
}
macro_rules! scalar { ($($ty:ty),*) => { $(impl Owned for $ty {
    fn decode<'de,D:Deserializer<'de>>(_: &mut Reservation,d:D)->Result<Self,D::Error> { serde::Deserialize::deserialize(d) }
})* }; }
scalar!(u8,u16,u64);
impl<T:Owned> Owned for Vec<T> {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error> {
        struct V<'a,T>(&'a mut Reservation,PhantomData<T>);
        impl<'de,T:Owned> Visitor<'de> for V<'_,T> {
            type Value=Vec<T>;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.write_str("admitted sequence")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error> {
                let mut out=Vec::<T>::new();
                loop {
                    if seq.size_hint()==Some(0) {break;}
                    if out.len()==out.capacity() {
                        let old=out.capacity();
                        let count=old.checked_mul(2).unwrap_or(usize::MAX).max(1);
                        let bytes=count.checked_mul(std::mem::size_of::<T>()).ok_or_else(||A::Error::custom(elasticsearch_transport::decode_admission_error("REST sequence capacity overflow")))?;
                        self.0.grow(bytes as u64,"REST sequence replacement").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                        out.try_reserve_exact(count-out.len()).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                        self.0.shrink((old*std::mem::size_of::<T>()) as u64);
                    }
                    match seq.next_element_seed(seed::<T>(self.0))? {Some(value)=>out.push(value),None=>break}
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V(memory,PhantomData))
    }
}
struct ByteString(Vec<u8>);
impl Owned for ByteString {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error>{
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_>{
            type Value=ByteString;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("admitted bytes")}
            fn visit_bytes<E:Error>(self,value:&[u8])->Result<Self::Value,E>{
                self.0.grow(value.len() as u64,"REST decoded body").map_err(|e|E::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut out=Vec::new();
                out.try_reserve_exact(value.len()).map_err(|e|E::custom(elasticsearch_transport::decode_admission_error(e)))?;
                out.extend_from_slice(value);Ok(ByteString(out))
            }
            fn visit_borrowed_bytes<E:Error>(self,value:&'de [u8])->Result<Self::Value,E>{self.visit_bytes(value)}
            fn visit_seq<A:SeqAccess<'de>>(self,seq:A)->Result<Self::Value,A::Error>{
                Vec::<u8>::decode(self.0,de::value::SeqAccessDeserializer::new(seq)).map(ByteString)
            }
        }
        d.deserialize_bytes(V(memory))
    }
}
impl<T:Owned> Owned for Option<T> {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error> {
        struct V<'a,T>(&'a mut Reservation,PhantomData<T>);
        impl<'de,T:Owned> Visitor<'de> for V<'_,T> {
            type Value=Option<T>;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.write_str("optional admitted field")}
            fn visit_none<E:Error>(self)->Result<Self::Value,E>{Ok(None)}
            fn visit_unit<E:Error>(self)->Result<Self::Value,E>{Ok(None)}
            fn visit_some<D:Deserializer<'de>>(self,d:D)->Result<Self::Value,D::Error>{T::decode(self.0,d).map(Some)}
        }
        d.deserialize_option(V(memory,PhantomData))
    }
}
impl Owned for (String,String) {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error> {
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_> {
            type Value=(String,String);
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("HTTP header pair")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let name=seq.next_element_seed(seed::<String>(self.0))?.ok_or_else(||A::Error::invalid_length(0,&self))?;
                let value=seq.next_element_seed(seed::<String>(self.0))?.ok_or_else(||A::Error::invalid_length(1,&self))?;
                Ok((name,value))
            }
        }
        d.deserialize_tuple(2,V(memory))
    }
}
macro_rules! owned_struct {
    ($ty:ident {$($field:ident : $ft:ty),* $(,)?}) => {
        impl Owned for $ty {
            fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error>{
                struct V<'a>(&'a mut Reservation);
                impl<'de> Visitor<'de> for V<'_> {
                    type Value=$ty;
                    fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.write_str(stringify!($ty))}
                    fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                        $(let $field=seq.next_element_seed(seed::<$ft>(self.0))?.ok_or_else(||A::Error::missing_field(stringify!($field)))?;)*
                        Ok($ty{$($field),*})
                    }
                    fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                        #[derive(serde::Deserialize)] #[serde(field_identifier)] #[allow(non_camel_case_types)]
                        enum Field {$($field,)* #[serde(other)] Ignore}
                        $(let mut $field=None;)*
                        while let Some(decoded_key)=map.next_key::<Field>()? {match decoded_key {
                            $(Field::$field=>{if $field.is_some(){return Err(A::Error::duplicate_field(stringify!($field)));}
                                $field=Some(map.next_value_seed(seed::<$ft>(self.0))?);},)*
                            Field::Ignore=>{let _:de::IgnoredAny=map.next_value()?;}
                        }}
                        Ok($ty{$($field:$field.ok_or_else(||A::Error::missing_field(stringify!($field)))?),*})
                    }
                }
                d.deserialize_struct(stringify!($ty),&[$(stringify!($field)),*],V(memory))
            }
        }
    };
}
owned_struct!(PlanData {indices:Vec<String>,query_json:Vec<u8>});
scalar!(AppSearchPlan,SharedDfsStats);
owned_struct!(DfsTerm {field:String,term:Vec<u8>,doc_freq:u64,total_term_freq:u64,field_doc_count:u64});
owned_struct!(DfsField {field:String,doc_count:u64,sum_total_term_freq:u64});
owned_struct!(DfsStats {terms:Vec<DfsTerm>,fields:Vec<DfsField>});
impl Owned for Dfs {
    fn decode<'de,D:Deserializer<'de>>(memory:&mut Reservation,d:D)->Result<Self,D::Error>{
        use serde::de::{EnumAccess,VariantAccess};
        #[derive(serde::Deserialize)] enum Variant {None,Collect,Use}
        struct V<'a>(&'a mut Reservation);
        impl<'de> Visitor<'de> for V<'_>{
            type Value=Dfs;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("DFS phase")}
            fn visit_enum<A:EnumAccess<'de>>(self,data:A)->Result<Self::Value,A::Error>{
                let (variant,value)=data.variant::<Variant>()?;
                match variant {
                    Variant::None=>{value.unit_variant()?;Ok(Dfs::None)},
                    Variant::Collect=>{value.unit_variant()?;Ok(Dfs::Collect)},
                    Variant::Use=>value.newtype_variant_seed(seed::<SharedDfsStats>(self.0)).map(Dfs::Use),
                }
            }
        }
        d.deserialize_enum("Dfs",&["None","Collect","Use"],V(memory))
    }
}

// Optional appended fields accept only the legacy EOF case. An admission
// rejection or invalid enum must never silently become the default phase.
fn tail<T:Default,E:Error>(result:Result<Option<T>,E>)->Result<T,E>{
    match result {Ok(v)=>Ok(v.unwrap_or_default()),Err(e) if e.to_string().contains("unexpected end of file")=>Ok(T::default()),Err(e)=>Err(e)}
}
impl<'de> serde::Deserialize<'de> for super::admitted::Request {
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=super::admitted::Request;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("admitted REST request")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let mut memory=elasticsearch_engine::memory::breakers().reserve_native(0,"REST decoded protocol prefix").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let method=seq.next_element_seed(seed::<String>(&mut memory))?.ok_or_else(||A::Error::missing_field("method"))?;
                let uri=seq.next_element_seed(seed::<String>(&mut memory))?.ok_or_else(||A::Error::missing_field("uri"))?;
                if elasticsearch_http::in_flight_policy(&method,&uri,true)!=elasticsearch_http::InFlight::Holds {
                    memory=memory.into_query("REST decoded request").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                }
                let headers=seq.next_element_seed(seed::<Vec<(String,String)>>(&mut memory))?.ok_or_else(||A::Error::missing_field("headers"))?;
                let body=seq.next_element_seed(seed::<ByteString>(&mut memory))?.ok_or_else(||A::Error::missing_field("body"))?.0;
                let hops=seq.next_element()?.ok_or_else(||A::Error::missing_field("hops"))?;
                let dfs=tail(seq.next_element_seed(seed::<Dfs>(&mut memory)))?;
                let app_plan=tail(seq.next_element_seed(seed::<Option<AppSearchPlan>>(&mut memory)))?;
                RestReq::from_reserved(method,uri,headers,body,hops,dfs,app_plan,memory).map(|value|super::admitted::Request{value,memory:()}).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(serde::Deserialize)] #[serde(field_identifier,rename_all="snake_case")]
                enum Field {Method,Uri,Headers,Body,Hops,Dfs,AppPlan,#[serde(other)]Ignore}
                let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"REST decoded request").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let (mut method,mut uri,mut headers,mut body,mut hops,mut dfs,mut app_plan)=(None,None,None,None,None,None,None);
                macro_rules! field {($slot:ident,$ty:ty)=>{{if $slot.is_some(){return Err(A::Error::duplicate_field(stringify!($slot)));}$slot=Some(map.next_value_seed(seed::<$ty>(&mut memory))?);}}}
                while let Some(key)=map.next_key::<Field>()?{match key {
                    Field::Method=>field!(method,String),Field::Uri=>field!(uri,String),Field::Headers=>field!(headers,Vec<(String,String)>),
                    Field::Body=>field!(body,ByteString),Field::Hops=>field!(hops,u8),Field::Dfs=>field!(dfs,Dfs),Field::AppPlan=>field!(app_plan,Option<AppSearchPlan>),
                    Field::Ignore=>{let _:de::IgnoredAny=map.next_value()?;}
                }}
                RestReq::from_reserved(
                    method.ok_or_else(||A::Error::missing_field("method"))?,uri.ok_or_else(||A::Error::missing_field("uri"))?,
                    headers.ok_or_else(||A::Error::missing_field("headers"))?,body.ok_or_else(||A::Error::missing_field("body"))?.0,
                    hops.ok_or_else(||A::Error::missing_field("hops"))?,dfs.unwrap_or_default(),app_plan.unwrap_or_default(),memory,
                ).map(|value|super::admitted::Request{value,memory:()}).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))
            }
        }
        d.deserialize_struct("RestReq",&["method","uri","headers","body","hops","dfs","app_plan"],V)
    }
}

pub(super) fn shared_dfs<'de,D:Deserializer<'de>>(d:D)->Result<SharedDfsStats,D::Error>{
 let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"DFS decode").map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
 let value=DfsStats::decode(&mut memory,d)?;
 SharedDfsStats::from_reserved(value,memory).map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))
}
pub(super) fn shared_plan<'de,D:Deserializer<'de>>(d:D)->Result<AppSearchPlan,D::Error>{
 let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"plan decode").map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
 let value=PlanData::decode(&mut memory,d)?;
 AppSearchPlan::from_reserved(value,memory).map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
    // These limits test wire allocation ownership independently of process RSS.
    fn fixture_breakers(bytes: u64) -> &'static Breakers {
        let budget = Breakers::leak_for_tests(bytes);
        budget.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        budget
    }
    #[test]
    fn decided_remote_error_survives_parent_pressure_but_success_does_not() {
        let b=Breakers::leak_for_tests(1<<20);
        b.set_real_memory_for_tests(Some(1<<20));
        let scope=RequestScope::new("remote response under pressure",b);
        let body=br#"{"error":{"type":"circuit_breaking_exception","reason":"original parent refusal"},"status":429}"#.to_vec();
        let headers=vec![("content-type".to_string(),"application/json".to_string())];
        let wire=bincode::serialize(&(429u16,&headers,&body)).unwrap();
        let response:super::super::RestResp=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();
        assert_eq!(response.status,429);assert_eq!(response.body.as_ref(),body.as_slice());
        assert_eq!(b.get(Kind::Request).used(),0);
        let successful=bincode::serialize(&(200u16,&headers,&body)).unwrap();
        assert!(scoped(Some(scope.clone()),||bincode::deserialize::<super::super::RestResp>(&successful)).is_err());
        drop(scope);drop(response);
        assert_eq!(b.get(Kind::Accounting).used(),0);
    }

    fn request() -> RestReq {
        RestReq::try_new("POST","/i/_search",&[("accept".into(),"application/json".into())],&vec![42;4096],1,
            Dfs::Use(SharedDfsStats::try_new(&DfsStats{terms:vec![DfsTerm{field:"f".into(),term:vec![7;1024],doc_freq:1,total_term_freq:2,field_doc_count:3}],fields:vec![]}).unwrap()),
            Some(AppSearchPlan::try_new(&["i".into()],&vec![b'x';1024]).unwrap())).unwrap()
    }
    #[test]
    fn forwarded_wire_admits_nested_fields_and_retains_owner_after_input_drop() {
        let original=request();let wire=bincode::serialize(&original).unwrap();
        let breakers=fixture_breakers(1<<20);let scope=RequestScope::new("REST wire",breakers);
        let decoded:super::super::admitted::Request=scoped(Some(scope.clone()),||bincode::deserialize(&wire)).unwrap();
        assert_eq!(bincode::serialize(&decoded.value).unwrap(),wire);
        let json=serde_json::to_string(&original).unwrap();
        let json_decoded:super::super::admitted::Request=scoped(Some(scope.clone()),||serde_json::from_str(&json)).unwrap();
        assert_eq!(bincode::serialize(&json_decoded.value).unwrap(),wire);
        drop(wire);drop(json);drop(original);
        assert!(breakers.get(Kind::Request).used()>=2*(4096+1024+1024));
        drop(decoded);drop(json_decoded);
        assert_eq!(breakers.get(Kind::Request).used(),0);
    }
    #[test]
    fn forwarded_appended_field_refusal_is_not_silently_defaulted() {
        let mut original=request();
        original.set_body(&[]).unwrap();
        original.set_dfs(Dfs::None).unwrap();
        let plan=AppSearchPlan::try_new(&original.app_plan.as_ref().unwrap().indices,&vec![b'x';65536]).unwrap();
        original.set_app_plan(Some(plan)).unwrap();
        let wire=bincode::serialize(&original).unwrap();
        let breakers=fixture_breakers(4096);let scope=RequestScope::new("REST refusal",breakers);
        let error=scoped(Some(scope),||bincode::deserialize::<super::super::admitted::Request>(&wire)).err().unwrap();
        assert!(error.to_string().contains("admission"),"{error}");
        assert_eq!(breakers.get(Kind::Request).used(),0);
    }
    #[test]
    fn legacy_request_without_appended_fields_remains_accepted() {
        #[derive(serde::Serialize)] struct Old {method:String,uri:String,headers:Vec<(String,String)>,body:Vec<u8>,hops:u8}
        let wire=bincode::serialize(&Old{method:"GET".into(),uri:"/i/_search".into(),headers:vec![],body:vec![],hops:1}).unwrap();
        let breakers=fixture_breakers(4096);let scope=RequestScope::new("legacy REST",breakers);
        let decoded:super::super::admitted::Request=scoped(Some(scope),||bincode::deserialize(&wire)).unwrap();
        assert_eq!(decoded.value.dfs,Dfs::None);assert!(decoded.value.app_plan.is_none());
        drop(decoded);assert_eq!(breakers.get(Kind::Request).used(),0);
    }
}

impl<'de> serde::Deserialize<'de> for super::RestResp {
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=super::RestResp;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("admitted REST response")}
            fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->Result<Self::Value,A::Error>{
                let status=seq.next_element::<u16>()?.ok_or_else(||A::Error::missing_field("status"))?;
                let breakers=elasticsearch_engine::memory::breakers();
                let mut hm=breakers.reserve_native(0,"REST decoded headers").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut bm=if status>=400 {breakers.reserve_native(0,"REST decoded error response")}else{breakers.reserve_request(0,"REST decoded response")}.map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let headers=seq.next_element_seed(seed::<Vec<(String,String)>>(&mut hm))?.ok_or_else(||A::Error::missing_field("headers"))?;
                let body=seq.next_element_seed(seed::<ByteString>(&mut bm))?.ok_or_else(||A::Error::missing_field("body"))?.0;
                super::RestResp::decoded(status,headers,hm,body,bm).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))
            }
            fn visit_map<A:MapAccess<'de>>(self,mut map:A)->Result<Self::Value,A::Error>{
                #[derive(serde::Deserialize)] #[serde(field_identifier,rename_all="snake_case")]
                enum Field{Status,Headers,Body,#[serde(other)]Ignore}
                let mut hm=elasticsearch_engine::memory::breakers().reserve_native(0,"REST decoded headers").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let mut bm=elasticsearch_engine::memory::breakers().reserve_native(0,"REST decoded response prefix").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                let (mut status,mut headers,mut body)=(None,None,None);
                while let Some(key)=map.next_key::<Field>()?{match key{
                    Field::Status=>{if status.is_some(){return Err(A::Error::duplicate_field("status"));}status=Some(map.next_value()?);},
                    Field::Headers=>{if headers.is_some(){return Err(A::Error::duplicate_field("headers"));}headers=Some(map.next_value_seed(seed::<Vec<(String,String)>>(&mut hm))?);},
                    Field::Body=>{if body.is_some(){return Err(A::Error::duplicate_field("body"));}body=Some(map.next_value_seed(seed::<ByteString>(&mut bm))?);},
                    Field::Ignore=>{let _:de::IgnoredAny=map.next_value()?;}
                }}
                let status=status.ok_or_else(||A::Error::missing_field("status"))?;
                if status<400 {
                    // Map field order is arbitrary. Admit ordinary successful
                    // output under the caller policy before exposing it.
                    let replacement=elasticsearch_engine::memory::breakers().reserve_request(bm.bytes(),"REST decoded response").map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
                    bm=replacement;
                }
                super::RestResp::decoded(status,headers.ok_or_else(||A::Error::missing_field("headers"))?,hm,body.ok_or_else(||A::Error::missing_field("body"))?.0,bm).map_err(|e|A::Error::custom(elasticsearch_transport::decode_admission_error(e)))
            }
        }
        d.deserialize_struct("RestResp",&["status","headers","body"],V)
    }
}

pub(super) fn shared_headers<'de,D:Deserializer<'de>>(d:D)->Result<super::SharedHeaders,D::Error>{
    let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"decoded shared headers").map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
    let value=Vec::<(String,String)>::decode(&mut memory,d)?;
    super::SharedHeaders::from_reserved(value,memory).map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))
}

pub(super) fn shared_text<'de,D:Deserializer<'de>>(d:D)->Result<super::SharedText,D::Error>{
    let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"decoded shared text").map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))?;
    let value=String::decode(&mut memory,d)?;
    super::SharedText::from_reserved(value,memory).map_err(|e|D::Error::custom(elasticsearch_transport::decode_admission_error(e)))
}
